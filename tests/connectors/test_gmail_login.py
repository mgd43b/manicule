"""``manicule connector login`` for Gmail: the installed-app flow, end to end.

Everything runs for real except Google's two hosts: the loopback listener is a socket on
``127.0.0.1``, the browser's visit is an HTTP request to it, and the token endpoint and
``getProfile`` are answered by :class:`~tests.connectors.fake_google_signin.FakeGoogleSignIn`
through the mock transport. What is asserted is what an operator and a reviewer care about: the
request Google is sent, what is refused, where the token ends up, and that it ends up nowhere
else.
"""

from __future__ import annotations

import io
import logging
import os
import socket
import stat
import threading
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors.errors import CredentialError, MailboxMismatchError
from manicule.connectors.gmail_login import (
    GMAIL_READONLY_SCOPE,
    FileSink,
    StreamSink,
    _challenge,  # pyright: ignore[reportPrivateUsage]
)
from manicule.core.errors import ConfigError
from tests.app.fakes import FakeBackend
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN
from tests.connectors.fake_google_signin import FakeGoogleSignIn, visit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ACCOUNT = "mail@example.org"


@pytest.fixture(autouse=True)
def client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.delenv("GMAIL_REFRESH_TOKEN", raising=False)


@pytest.fixture
def google() -> FakeGoogleSignIn:
    return FakeGoogleSignIn(address=ACCOUNT)


@pytest.fixture
def service() -> ApplicationService:
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
            "notes": ConnectorSettings(type="filesystem", options={"root": "/srv/notes"}),
            "off": ConnectorSettings(
                type="gmail",
                enabled=False,
                options={
                    "account": ACCOUNT,
                    "client_id": "1234-example.apps.googleusercontent.com",
                },
            ),
        }
    )
    return ApplicationService(backend)


def browser(google: FakeGoogleSignIn, **choices: Any) -> Callable[[str], bool]:
    """A browser that sends the person's choice to the loopback socket the command opened."""
    return lambda url: google.browse(url, **choices)


async def login(
    service: ApplicationService,
    google: FakeGoogleSignIn,
    sink: Any,
    *,
    open_browser: Callable[[str], object] | None = None,
    timeout_seconds: float = 20.0,
    announce: Callable[[str], None] | None = None,
) -> Any:
    return await service.connector_login_gmail(
        "mail",
        sink=sink,
        timeout_seconds=timeout_seconds,
        announce=announce,
        open_browser=open_browser or browser(google),
        transport=google.transport,
    )


async def test_the_whole_flow_writes_the_token_to_a_private_file(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    target = tmp_path / "gmail-token"

    result = await login(service, google, FileSink(target))

    assert target.read_text() == REFRESH_TOKEN + "\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert (result.name, result.account, result.scope) == ("mail", ACCOUNT, GMAIL_READONLY_SCOPE)
    assert REFRESH_TOKEN not in result.model_dump_json()

    (asked,) = google.authorizations
    assert asked["client_id"] == "1234-example.apps.googleusercontent.com"
    assert asked["scope"] == GMAIL_READONLY_SCOPE
    assert (asked["access_type"], asked["prompt"]) == ("offline", "consent")
    assert asked["code_challenge_method"] == "S256"
    assert asked["login_hint"] == ACCOUNT
    assert asked["redirect_uri"].startswith("http://127.0.0.1:")

    # One code exchange, then exactly the verification the issue asks for: one refresh and one
    # getProfile, made with the token that was just minted.
    (exchange,) = google.exchanges
    assert exchange["redirect_uri"] == asked["redirect_uri"]
    assert exchange["client_secret"] == CLIENT_SECRET
    assert google.refreshes == 1
    assert [str(r.url).rsplit("/", 1)[-1] for r in google.requests].count("profile") == 1


def test_the_pkce_challenge_is_the_s256_of_the_verifier() -> None:
    """The worked example in RFC 7636, appendix B."""
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert _challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


async def test_a_redirect_with_the_wrong_state_is_refused_and_nothing_is_exchanged(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    target = tmp_path / "token"

    with pytest.raises(CredentialError, match="state"):
        await login(service, google, FileSink(target), open_browser=browser(google, state="forged"))

    assert google.visits == [400]  # the socket itself refused the forged redirect
    assert google.exchanges == []
    assert not target.exists()


async def test_declining_in_the_browser_ends_the_sign_in_without_a_token(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    target = tmp_path / "token"

    with pytest.raises(CredentialError, match="declined"):
        await login(service, google, FileSink(target), open_browser=browser(google, approve=False))

    assert google.exchanges == []
    assert not target.exists()


async def test_a_token_for_another_mailbox_is_refused_and_written_nowhere(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    """The browser was signed in to someone else: a perfectly good token for the wrong mailbox."""
    google.address = "someone-else@example.org"
    target = tmp_path / "token"

    with pytest.raises(MailboxMismatchError) as caught:
        await login(service, google, FileSink(target))

    assert not target.exists()
    message = str(caught.value)
    assert "example.org" not in message
    assert REFRESH_TOKEN not in message


@pytest.mark.parametrize(
    "scope",
    [
        f"{GMAIL_READONLY_SCOPE} https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/userinfo.email",
        "",
    ],
)
async def test_a_token_that_is_not_exactly_read_only_is_discarded(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path, scope: str
) -> None:
    google.granted_scope = scope
    target = tmp_path / "token"

    with pytest.raises(CredentialError, match=r"gmail\.readonly"):
        await login(service, google, FileSink(target))

    assert not target.exists()


async def test_google_issuing_no_refresh_token_is_reported_for_what_it_is(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    google.issues_refresh_token = False

    with pytest.raises(CredentialError, match="no refresh token"):
        await login(service, google, FileSink(tmp_path / "token"))


async def test_waiting_for_a_sign_in_that_never_comes_times_out(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    with pytest.raises(CredentialError, match="no sign-in arrived"):
        await login(
            service,
            google,
            FileSink(tmp_path / "token"),
            open_browser=lambda _url: True,
            timeout_seconds=0.3,
        )


async def test_the_listener_ignores_strays_and_answers_only_the_real_redirect(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    """A favicon fetch, a probe with the wrong ``Host`` and a POST do not end the sign-in."""

    def raw(port: int, request: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            conn.sendall(request)
            return conn.recv(4096)

    answers: list[bytes] = []

    def noisy(url: str) -> bool:
        redirect = google.consent(url)
        port = int(redirect.split(":")[2].split("/")[0])
        host = f"Host: 127.0.0.1:{port}\r\n\r\n".encode()
        answers.extend(
            raw(port, request)[:12]
            for request in (
                b"GET /favicon.ico HTTP/1.1\r\n" + host,
                b"GET / HTTP/1.1\r\n" + host,
                b"POST /?code=x HTTP/1.1\r\n" + host,
                # A name that resolves here is still not this socket's address.
                b"GET /?code=x&state=y HTTP/1.1\r\nHost: evil.example\r\n\r\n",
            )
        )
        google.visits.append(visit(redirect))
        return True

    target = tmp_path / "token"
    await login(service, google, FileSink(target), open_browser=noisy)

    assert answers == [b"HTTP/1.1 404", b"HTTP/1.1 404", b"HTTP/1.1 405", b"HTTP/1.1 400"]
    assert google.visits == [200]
    assert target.read_text() == REFRESH_TOKEN + "\n"


async def test_the_address_to_open_by_hand_is_announced_with_nothing_secret_in_it(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    said: list[str] = []

    await login(service, google, FileSink(tmp_path / "token"), announce=said.append)

    text = "\n".join(said)
    assert "https://accounts.google.com/o/oauth2/v2/auth?" in text
    assert CLIENT_SECRET not in text
    assert REFRESH_TOKEN not in text
    assert "code_verifier" not in text


# --- where the token may go ---------------------------------------------------------------------


async def test_a_destination_that_cannot_take_the_token_is_found_before_the_browser_opens(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    existing = tmp_path / "token"
    existing.write_text("keep me")
    opened: list[str] = []

    with pytest.raises(ConfigError, match="already exists"):
        await login(
            service,
            google,
            FileSink(existing),
            open_browser=lambda url: opened.append(url) or True,
        )

    assert opened == []
    assert existing.read_text() == "keep me"


async def test_overwriting_replaces_the_file_and_leaves_no_staging_file_behind(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    directory = tmp_path / "secrets"
    directory.mkdir()
    existing = directory / "token"
    existing.write_text("old")
    existing.chmod(0o644)

    await login(service, google, FileSink(existing, overwrite=True))

    assert existing.read_text() == REFRESH_TOKEN + "\n"
    assert stat.S_IMODE(existing.stat().st_mode) == 0o600
    assert [p.name for p in directory.iterdir()] == ["token"]


def test_a_file_is_private_whatever_the_umask_says(tmp_path: Path) -> None:
    from pydantic import SecretStr  # noqa: PLC0415

    old = os.umask(0)
    try:
        FileSink(tmp_path / "token").write(SecretStr("t"))
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "token").stat().st_mode) == 0o600


def test_a_symbolic_link_is_never_followed(tmp_path: Path) -> None:
    from pydantic import SecretStr  # noqa: PLC0415

    victim = tmp_path / "victim"
    victim.write_text("untouched")
    link = tmp_path / "token"
    link.symlink_to(victim)

    with pytest.raises(ConfigError, match="already exists"):
        FileSink(link).check()
    with pytest.raises(FileExistsError):
        FileSink(link).write(SecretStr("t"))

    assert victim.read_text() == "untouched"


def test_a_missing_directory_is_refused_up_front(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a directory"):
        FileSink(tmp_path / "absent" / "token").check()


class _Stream(io.StringIO):
    def __init__(self, *, terminal: bool) -> None:
        super().__init__()
        self._terminal = terminal

    @override
    def isatty(self) -> bool:
        return self._terminal


async def test_a_terminal_is_refused_before_the_browser_opens(
    service: ApplicationService, google: FakeGoogleSignIn
) -> None:
    screen = _Stream(terminal=True)
    opened: list[str] = []

    with pytest.raises(ConfigError, match="terminal"):
        await login(
            service, google, StreamSink(screen), open_browser=lambda u: opened.append(u) or True
        )

    assert opened == []
    assert screen.getvalue() == ""


async def test_a_pipe_receives_the_token_and_only_the_token(
    service: ApplicationService, google: FakeGoogleSignIn
) -> None:
    pipe = _Stream(terminal=False)

    await login(service, google, StreamSink(pipe))

    assert pipe.getvalue() == REFRESH_TOKEN + "\n"


# --- who may ask --------------------------------------------------------------------------------


async def test_only_a_gmail_source_that_is_enabled_can_sign_in(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    sink = FileSink(tmp_path / "token")
    with pytest.raises(ConfigError, match="filesystem"):
        await service.connector_login_gmail("notes", sink=sink, transport=google.transport)
    with pytest.raises(Exception, match="enabled = false"):
        await service.connector_login_gmail("off", sink=sink, transport=google.transport)
    with pytest.raises(Exception, match="no connector named"):
        await service.connector_login_gmail("absent", sink=sink, transport=google.transport)


async def test_a_missing_client_secret_names_the_variable_and_does_not_open_a_browser(
    service: ApplicationService,
    google: FakeGoogleSignIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GMAIL_CLIENT_SECRET")
    opened: list[str] = []

    with pytest.raises(ConfigError, match=r"GMAIL_CLIENT_SECRET"):
        await login(
            service,
            google,
            FileSink(tmp_path / "token"),
            open_browser=lambda u: opened.append(u) or True,
        )

    assert opened == []


# --- the token never reaches a log --------------------------------------------------------------


async def test_no_log_record_carries_the_token_or_the_code(
    service: ApplicationService,
    google: FakeGoogleSignIn,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")

    await login(service, google, FileSink(tmp_path / "token"))

    assert caplog.records, "a log with nothing in it proves nothing"
    logged = "\n".join(f"{r.name} {r.getMessage()}" for r in caplog.records)
    code = google.exchanges[0]["code"]
    for secret in (REFRESH_TOKEN, CLIENT_SECRET, code, "access-token-sentinel"):
        assert secret not in logged
    assert "example.org" not in logged


async def test_no_failure_message_carries_a_secret(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    google.fail_refresh(400, "invalid_grant")  # the verification refresh is refused
    with pytest.raises(CredentialError) as caught:
        await login(service, google, FileSink(tmp_path / "token"))
    assert REFRESH_TOKEN not in str(caught.value)
    assert CLIENT_SECRET not in str(caught.value)
    assert "sentinel-prose" not in str(caught.value)


async def test_a_browser_that_stays_open_does_not_hold_the_sign_in_up(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    """A console browser does not return until the person quits it; the token must not wait."""
    quit_browser = threading.Event()

    def lingering(url: str) -> bool:
        google.browse(url)
        quit_browser.wait(timeout=10)
        return True

    target = tmp_path / "token"
    try:
        await login(service, google, FileSink(target), open_browser=lingering)
    finally:
        quit_browser.set()

    assert target.read_text() == REFRESH_TOKEN + "\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
async def test_a_directory_that_cannot_be_written_is_found_before_the_browser_opens(
    service: ApplicationService, google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    opened: list[str] = []
    try:
        with pytest.raises(ConfigError, match="not writable"):
            await login(
                service,
                google,
                FileSink(locked / "token"),
                open_browser=lambda u: opened.append(u) or True,
            )
    finally:
        locked.chmod(0o700)

    assert opened == []
