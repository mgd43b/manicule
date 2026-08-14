"""The session crosses to the server, is used there, and appears nowhere else.

This is now the **only** place a live Confluence session exists in a running manicule, so every
surface it can reach is a new surface. The last section asserts over each of them by name rather
than trusting that nothing prints a credential: the value is a string nobody chose, so a test can
look for it in whatever a surface produced and be sure the answer means something.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from manicule.app import control
from manicule.app.served import ControlHandler
from manicule.app.service import ApplicationService
from manicule.cli import proxy
from manicule.connectors.credentials import BrowserSession, credential_for
from manicule.connectors.sessions import SessionVault, load_session
from manicule.core.errors import ConfigError
from tests.app.fakes import FakeBackend
from tests.connectors import support

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from manicule.connectors.config import ConfluenceConfig

SITE = "https://wiki.example.test"

SENTINEL = "sentinel-session-value-0f9a3b"
"""The cookie value every assertion below hunts for.

A string that appears nowhere else in the repository, so finding it in a rendered payload, a
log line or an exception message means it came from the session and not from a fixture that
happened to share a word.
"""


@pytest.fixture
def socket_for() -> Iterator[Callable[[], Path]]:
    made: list[Path] = []

    def build() -> Path:
        path = control.socket_path(Path(f"/manicule-suite/{uuid.uuid4()}"))
        made.append(path)
        return path

    yield build
    for path in made:
        path.unlink(missing_ok=True)


def a_session(*, value: str = SENTINEL, account: str = "sync.user") -> BrowserSession:
    return BrowserSession(
        base_url=SITE,
        account=account,
        captured_at=datetime.now(tz=UTC),
        cookies={"JSESSIONID": SecretStr(value)},
    )


def sso_config(**overrides: object) -> ConfluenceConfig:
    """A Server instance whose identity provider left it with no tokens to offer.

    Built through the suite's own helper rather than spelled out here, so a change to how a
    browser-session source is configured reaches this file too.
    """
    return support.sso_config(SITE, **overrides)


def a_server(path: Path, vault: SessionVault) -> control.ControlServer:
    return control.ControlServer(path, ControlHandler(ApplicationService(FakeBackend()), vault))


# --- the hand-off ------------------------------------------------------------------------------


async def test_a_session_handed_to_the_server_is_usable_by_a_sync_it_did_not_capture(
    socket_for: Callable[[], Path],
) -> None:
    """The whole arrangement in one test: captured in one process, used in another.

    ``connector login`` runs where the person is, because ``--browser`` opens a window. The
    syncs run in the server. What joins them is this hand-off, and what proves it worked is that
    :func:`~manicule.connectors.credentials.credential_for` — the plugin factory's own call,
    which is what every sync goes through — builds a working credential out of the server's
    vault afterwards.
    """
    path = socket_for()
    vault = SessionVault()
    server = a_server(path, vault)
    await server.start()
    try:
        await proxy.HandoverStore(path).save(a_session())
    finally:
        await server.aclose()

    credential = credential_for(sso_config(), store=vault)

    assert credential.account() == "sync.user"
    assert credential.authorize().headers["Cookie"] == f"JSESSIONID={SENTINEL}", (
        "the session reached the server but does not authenticate a request there"
    )


async def test_the_capturing_process_keeps_no_copy(socket_for: Callable[[], Path]) -> None:
    """A command-line process is not a credential store and must not quietly become one.

    ``HandoverStore.load`` answering ``None`` is the assertion: a short-lived process holding a
    live corporate session in memory is most of what was wrong with the arrangements this
    replaced, and it would be invisible.
    """
    path = socket_for()
    vault = SessionVault()
    server = a_server(path, vault)
    await server.start()
    store = proxy.HandoverStore(path)
    try:
        await store.save(a_session())
    finally:
        await server.aclose()

    assert store.load(SITE) is None
    assert vault.load(SITE) is not None, "the server is where it went"


async def test_a_session_is_gone_after_the_server_restarts(
    socket_for: Callable[[], Path],
) -> None:
    """The expected path rather than a failure, and the message has to say so.

    A restart is how a session ends, because there is nowhere else for one to live. What must
    not happen is a refusal that reads like a fault — so this checks the message names both the
    command that fixes it and the reason there is nothing to fix.
    """
    path = socket_for()
    first = SessionVault()
    server = a_server(path, first)
    await server.start()
    try:
        await proxy.HandoverStore(path).save(a_session())
    finally:
        await server.aclose()

    restarted = SessionVault()

    assert first.load(SITE) is not None
    assert load_session(sso_config(), store=restarted) is None
    with pytest.raises(ConfigError) as raised:
        credential_for(sso_config(), store=restarted)
    message = str(raised.value)
    assert "connector login" in message, message
    assert "manicule serve" in message, message
    assert "expected path rather than a fault" in message, message


async def test_forgetting_reaches_the_server_that_holds_it(
    socket_for: Callable[[], Path],
) -> None:
    """``--forget`` has to remove the credential where it actually is, not where it is not."""
    path = socket_for()
    vault = SessionVault()
    server = a_server(path, vault)
    await server.start()
    store = proxy.HandoverStore(path)
    try:
        await store.save(a_session())
        assert await store.forget(SITE) is True
        assert await store.forget(SITE) is False, "forgetting twice says so the second time"
    finally:
        await server.aclose()

    assert vault.load(SITE) is None


async def test_a_handover_with_no_cookies_is_refused_rather_than_held(
    socket_for: Callable[[], Path],
) -> None:
    """A session with no cookies authenticates as nobody, and holding one would mean every
    later sync failing against a credential that looks present."""
    path = socket_for()
    vault = SessionVault()
    server = a_server(path, vault)
    await server.start()
    try:
        answered = await control.connect(
            path,
            control.Handover(
                base_url=SITE, account="sync.user", captured_at=datetime.now(tz=UTC).isoformat()
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    assert answered["ok"] is False
    assert vault.load(SITE) is None


async def test_the_capture_time_the_client_proved_is_the_one_the_server_measures_from(
    socket_for: Callable[[], Path],
) -> None:
    """``session_max_age_hours`` measures from capture, so a hand-off that restamped the time
    would silently extend every session's life by however long the hand-off took — and, worse,
    would reset the ceiling on a session re-sent by a retry.
    """
    path = socket_for()
    vault = SessionVault()
    captured = datetime.now(tz=UTC) - timedelta(hours=3)
    server = a_server(path, vault)
    await server.start()
    try:
        await proxy.HandoverStore(path).save(
            BrowserSession(
                base_url=SITE,
                account="sync.user",
                captured_at=captured,
                cookies={"JSESSIONID": SecretStr(SENTINEL)},
            )
        )
    finally:
        await server.aclose()

    held = vault.load(SITE)
    assert held is not None
    assert held.captured_at == captured

    from manicule.connectors.errors import SessionExpiredError  # noqa: PLC0415

    with pytest.raises(SessionExpiredError):
        credential_for(sso_config(session_max_age_hours=2.0), store=vault)


# --- everywhere it must not appear -------------------------------------------------------------


async def test_no_session_value_reaches_the_server_s_reply(
    socket_for: Callable[[], Path],
) -> None:
    """The reply is the first new surface, and it is one a terminal keeps in its scrollback.

    There is nothing a caller needs from the value it just sent, so the acknowledgment names the
    instance and the account and carries no cookie, no length and no digest of one.
    """
    path = socket_for()
    server = a_server(path, SessionVault())
    await server.start()
    try:
        answered = await control.connect(
            path,
            control.Handover(
                base_url=SITE,
                account="sync.user",
                captured_at=datetime.now(tz=UTC).isoformat(),
                cookies={"JSESSIONID": SecretStr(SENTINEL)},
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    assert SENTINEL not in json.dumps(answered)
    assert answered["ok"] is True


def test_no_session_value_reaches_a_credential_s_own_messages() -> None:
    """Every refusal this connector raises names a URL and several name the credential's
    description. None of them may name the credential."""
    credential = credential_for(sso_config(), store=_vault_holding(a_session()))

    assert SENTINEL not in repr(credential)
    assert SENTINEL not in credential.describe()
    assert SENTINEL not in credential.renewal()
    assert SENTINEL not in str(credential.account())


def test_no_session_value_reaches_an_expiry_refusal() -> None:
    """The one message that is *about* the session, and therefore the one most likely to quote
    it. It names the age, the ceiling and the setting, and not the value."""
    from manicule.connectors.errors import SessionExpiredError  # noqa: PLC0415

    old = BrowserSession(
        base_url=SITE,
        account="sync.user",
        captured_at=datetime.now(tz=UTC) - timedelta(hours=9),
        cookies={"JSESSIONID": SecretStr(SENTINEL)},
    )
    with pytest.raises(SessionExpiredError) as raised:
        credential_for(sso_config(session_max_age_hours=1.0), store=_vault_holding(old))

    assert SENTINEL not in str(raised.value)
    assert "session_max_age_hours" in str(raised.value)


def test_no_session_value_reaches_the_vault_s_own_rendering() -> None:
    """A vault is the object a future diagnostic, report or crash dump would reach for."""
    vault = _vault_holding(a_session())

    assert SENTINEL not in repr(vault)
    assert SENTINEL not in str(vault)
    assert SENTINEL not in vault.describe()


def test_no_session_value_reaches_a_frame_that_will_not_parse() -> None:
    """The socket's error path, which is the surface a validation failure would leak through.

    pydantic quotes the offending input, which is right everywhere else and wrong for the one
    frame that carries a live session. Asserted here as well as in ``test_control.py`` because
    the two tests are about different things: that one is about the parser, and this one is
    about the credential.
    """
    line = json.dumps(
        {
            "kind": "handover",
            "base_url": SITE,
            "account": "sync.user",
            "captured_at": "not-a-time",
            "cookies": {"JSESSIONID": [SENTINEL]},
        }
    ).encode()
    with pytest.raises(control.ProtocolError) as raised:
        control.read_request(line)

    assert SENTINEL not in str(raised.value)


async def test_no_session_value_reaches_a_report_a_person_reads(
    socket_for: Callable[[], Path],
) -> None:
    """``connector login`` renders a payload and ``--json`` prints it. Neither may carry it.

    The rendered form is checked as well as the model, because a payload can be clean and a
    renderer can still reach past it — and the rendered form is the one that ends up in a
    terminal somebody screenshots.
    """
    from manicule.app import results as r  # noqa: PLC0415
    from manicule.cli import render  # noqa: PLC0415

    path = socket_for()
    server = a_server(path, SessionVault())
    await server.start()
    try:
        store = proxy.HandoverStore(path)
        await store.save(a_session())
    finally:
        await server.aclose()

    payload = r.ConnectorSignedIn(
        name="handbook",
        base_url=SITE,
        account="sync.user",
        captured_at=datetime.now(tz=UTC).isoformat(),
        expires_at=(datetime.now(tz=UTC) + timedelta(hours=8)).isoformat(),
        stored_in=store.describe(),
    )
    console = render.console()
    with console.capture() as captured:
        render.render(console, payload)

    assert SENTINEL not in payload.model_dump_json()
    assert SENTINEL not in captured.get()
    assert "server" in payload.stored_in, "the report says where the session actually went"


def test_no_session_value_survives_a_traceback_through_the_vault() -> None:
    """A crash dump is a surface too, and it is the one nobody designs.

    ``SecretStr`` is what makes this structural rather than careful: a frame's locals render the
    wrapper, so a traceback printed by anything — pytest, a logger, an unhandled exception
    handler — carries asterisks.
    """
    import traceback  # noqa: PLC0415

    vault = _vault_holding(a_session())
    try:
        session = vault.load(SITE)
        assert session is not None
        msg = "something failed while a session was in scope"
        raise RuntimeError(msg)  # noqa: TRY301 - the traceback is the subject
    except RuntimeError:
        rendered = traceback.format_exc()

    assert SENTINEL not in rendered


def _vault_holding(session: BrowserSession) -> SessionVault:
    vault = SessionVault()
    asyncio.run(vault.save(session))
    return vault
