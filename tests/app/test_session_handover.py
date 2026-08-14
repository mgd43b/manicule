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
from manicule.config.settings import ConnectorSettings
from manicule.connectors.credentials import BrowserSession, credential_for
from manicule.connectors.sessions import SESSIONS, SessionVault, load_session
from manicule.core.errors import ConfigError
from tests.app.fakes import FakeBackend
from tests.connectors import support

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from manicule.app.results import Check
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
        # The data directory is recorded as well as the socket, because `socket_path` is a digest
        # and cannot be inverted — and one test has to point a service's `data_dir` at whatever
        # produced the socket it is driving.
        directory = Path(f"/manicule-suite/{uuid.uuid4()}")
        _SUITE_DIRECTORIES.append(directory)
        path = control.socket_path(directory)
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


async def test_a_capture_time_with_no_offset_is_read_as_utc(
    socket_for: Callable[[], Path],
) -> None:
    """A frame carrying a naive timestamp must not become a crash at the far end of the system.

    ``BrowserSession.from_json`` normalized this and went with the keychain; the socket is now
    where a hand-built or older frame arrives. It matters more here than it did there:
    ``session_max_age_hours`` compares the capture time against an aware ``now()``, so a naive
    value is a ``TypeError`` out of ``authorize()`` on the first request of the next sync —
    a crash a long way from the field that caused it.
    """
    path = socket_for()
    vault = SessionVault()
    server = a_server(path, vault)
    await server.start()
    try:
        answered = await control.connect(
            path,
            control.Handover(
                base_url=SITE,
                account="sync.user",
                captured_at="2026-08-14T10:00:00",
                cookies={"JSESSIONID": SecretStr(SENTINEL)},
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()

    assert answered["ok"] is True
    held = vault.load(SITE)
    assert held is not None
    assert held.captured_at == datetime(2026, 8, 14, 10, 0, tzinfo=UTC)
    # The check that would have raised: an age comparison against an aware clock.
    credential_for(sso_config(session_max_age_hours=100000.0), store=vault).authorize()


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


async def test_no_session_value_reaches_the_reply_that_says_what_is_held(
    socket_for: Callable[[], Path],
) -> None:
    """The one new control frame, and it is the frame whose *purpose* is to talk about sessions.

    ``Handover``'s acknowledgment has been checked since #139. This is the other direction — a
    client asking what the server holds — and it is the more dangerous of the two, because the
    honest implementation reaches into the vault and the tempting one hands back what it found.

    **Found by disabling the guard and watching nothing go red.** Every other assertion in this
    file searches something a *surface* rendered; none of them touched this reply, so a listing
    that carried the sessions themselves passed the whole suite.
    """
    path = socket_for()
    vault = SessionVault()
    await vault.save(a_session())
    server = a_server(path, vault)
    await server.start()
    try:
        answered = await control.connect(path, control.Held(), on_progress=lambda _: None)
    finally:
        await server.aclose()

    assert answered["ok"] is True, answered
    assert SENTINEL not in json.dumps(answered)
    # And it did report the session, so the search above looked at a reply that had one to leak.
    data = answered["data"]
    assert isinstance(data, dict)
    assert data["held"] == [{"base_url": SITE, "account": "sync.user"}], data


async def test_no_session_value_reaches_a_tool_result_on_the_network_surface() -> None:
    """The MCP endpoint mounted on the HTTP port, which is a surface that did not exist before.

    #139 asserted this over the control socket's reply, the vault's rendering, a traceback and a
    refusal. None of those statements carries to a *served* MCP tool, because a tool's result is
    built by a different path — so ``doctor``, which is the tool that has anything to do with
    sessions at all, is called for real and its whole envelope is searched.

    ``doctor`` deliberately rather than ``search``: it is the read-only tool that now reports
    whether a session is held, so it is the one with a reason to be near the value. A tool with no
    connection to sessions would be a weaker place to look.
    """
    from tests.api.live import mounted  # noqa: PLC0415 - a socket-shaped fixture
    from tests.api.support import backend_with_a_document  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    backend.settings.connectors["handbook"] = _confluence_source()
    await SESSIONS.save(a_session())
    try:
        async with mounted(backend) as client:
            answered = (await client.call_tool("doctor", {})).structured_content or {}
    finally:
        await SESSIONS.forget(SITE)

    assert answered["ok"] is True, answered
    assert SENTINEL not in json.dumps(answered)
    # And the check did run against the held session, so the search above looked somewhere the
    # value could have been rather than at an envelope that never mentioned sessions.
    checks = {check["name"]: check for check in answered["data"]["checks"]}
    assert checks["sessions"]["state"] == "ok", checks["sessions"]
    assert "sync.user" in checks["sessions"]["detail"], checks["sessions"]


async def test_no_session_value_reaches_a_failed_call_on_the_network_surface() -> None:
    """The failure path, which is where a value leaks if it leaks anywhere.

    A success is composed by a payload model somebody wrote; a failure is composed from an
    exception's own message, and an exception is the thing most likely to have been raised near
    a credential with the credential in scope. So a tool is called in a way that fails while a
    session is held, and the whole refused envelope is searched.
    """
    from tests.api.live import mounted  # noqa: PLC0415 - a socket-shaped fixture
    from tests.api.support import backend_with_a_document  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    backend.settings.connectors["handbook"] = _confluence_source()
    await SESSIONS.save(a_session())
    try:
        async with mounted(backend) as client:
            refused = (
                await client.call_tool("document_get", {"document_id": "no-such-document"})
            ).structured_content or {}
    finally:
        await SESSIONS.forget(SITE)

    assert refused["ok"] is False, "the call succeeded, so no failure was searched"
    assert SENTINEL not in json.dumps(refused)


async def test_no_session_value_reaches_an_http_response() -> None:
    """The other surface on the same port, checked the same way and for the same reason.

    The browser surface renders ``doctor`` as a page and the JSON API returns it as an envelope.
    Both are new places for a credential to appear that #139's assertions say nothing about, and
    a page is the one somebody screenshots.
    """
    from tests.api.support import backend_with_a_document, client_for  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    backend.settings.connectors["handbook"] = _confluence_source()
    await SESSIONS.save(a_session())
    try:
        with client_for(backend) as client:
            api = client.get("/api/v1/health")
            page = client.get("/ui/health")
    finally:
        await SESSIONS.forget(SITE)

    assert SENTINEL not in api.text
    assert SENTINEL not in page.text


def _confluence_source() -> ConnectorSettings:
    """A Confluence source that authenticates with a browser session, so the check has work."""
    return ConnectorSettings.model_validate(
        {
            "type": "confluence",
            "options": {"base_url": SITE, "deployment": "server", "auth": "browser_session"},
        }
    )


# --- what an operator is told after a restart ----------------------------------------------------


async def test_doctor_reports_the_session_the_server_holds(
    socket_for: Callable[[], Path],
) -> None:
    """A ``doctor`` typed at a terminal reads the *server's* sessions, not its own.

    The whole difficulty of this check in one test: the process running ``doctor`` holds nothing
    and never will, so an answer taken from its own vault would say "no session" for ever. The
    question goes over the control socket to the process that has the answer, and the account it
    names is proof it went there rather than being inferred.
    """
    path = socket_for()
    vault = SessionVault()
    await vault.save(a_session(account="sync.user"))
    server = a_server(path, vault)
    await server.start()
    try:
        check = await _sessions_check(path, held_here=False)
    finally:
        await server.aclose()

    assert check.state == "ok", check.detail
    assert "sync.user" in check.detail, check.detail
    assert check.facts["read_from"] == "the running server, over its control socket"


async def test_doctor_reports_a_server_that_has_no_session_and_names_the_way_back(
    socket_for: Callable[[], Path],
) -> None:
    """The state a restart leaves, which is the whole reason this check exists.

    Three things have to be in the sentence, and each is a different mistake to avoid. It must
    name the **source**, because an operator with four of them has to know which. It must say the
    instance was **not contacted**, because otherwise this is indistinguishable from an outage.
    And it must name the **command**, because "sign in again" is not an instruction.
    """
    path = socket_for()
    server = a_server(path, SessionVault())
    await server.start()
    try:
        check = await _sessions_check(path, held_here=False)
    finally:
        await server.aclose()

    assert check.state == "degraded"
    assert "'handbook'" in check.detail, check.detail
    assert "nothing has been asked of it" in check.detail, check.detail
    assert check.remedy == "manicule connector login handbook --browser"
    assert check.facts["missing"] == ["handbook"]


async def test_doctor_says_when_there_is_no_server_at_all_rather_than_no_session(
    socket_for: Callable[[], Path],
) -> None:
    """The two states an operator would otherwise conflate, and they have different remedies.

    No server means nothing is scheduled and nothing holds a credential; the fix is
    ``manicule serve``. A server with no session means the schedule is running and failing; the
    fix is a browser. Telling somebody to sign in when there is no server to sign in *to* sends
    them to a refusal.
    """
    path = socket_for()
    assert not path.exists(), "this test needs a socket nothing is listening on"

    check = await _sessions_check(path, held_here=False)

    assert check.state == "degraded"
    assert check.facts["server"] is False
    assert check.remedy == "manicule serve"
    assert "manicule serve" in check.detail


async def test_doctor_inside_the_server_reads_the_vault_in_front_of_it() -> None:
    """The other vantage point: ``doctor`` over MCP or an HTTP route runs *in* the server.

    There the vault is the one the syncs read, so it is read directly rather than asked for over
    a socket the process would be connecting to itself on. The two are told apart by which vault
    holds something, which needs no flag: nothing but a server ever holds a session.
    """
    await SESSIONS.save(a_session(account="held.here"))
    try:
        check = await _sessions_check(None, held_here=True)
    finally:
        await SESSIONS.forget(SITE)

    assert check.state == "ok"
    assert "held.here" in check.detail
    assert check.facts["read_from"] == "this process, which holds them"


async def _sessions_check(path: Path | None, *, held_here: bool) -> Check:
    """Run ``doctor`` and hand back its ``sessions`` check.

    ``path`` names the socket the check should consult, which is normally derived from the data
    directory. It is steered here by pointing the settings at the directory that socket was
    derived from, rather than by patching the function that derives it — a test that replaced the
    derivation would not be testing the derivation.
    """
    del held_here  # the vault decides; this parameter documents which case the caller set up
    backend = FakeBackend()
    service = ApplicationService(backend)
    service.settings.connectors.clear()
    service.settings.connectors["handbook"] = _confluence_source()
    if path is not None:
        service.settings.data_dir = _directory_for(path)
    diagnosis = await service.doctor()
    return next(check for check in diagnosis.checks if check.name == "sessions")


def _directory_for(path: Path) -> Path:
    """The data directory whose socket is ``path``, recovered from the fixture that made it.

    ``socket_path`` is a digest, so it cannot be inverted. The fixture builds sockets from
    ``/manicule-suite/<uuid>``, and this reconstructs that: the same input gives the same digest,
    which is the whole property the naming relies on.
    """
    for candidate in _SUITE_DIRECTORIES:
        if control.socket_path(candidate) == path:
            return candidate
    msg = f"no suite data directory produces {path}"  # pragma: no cover - the fixture records all
    raise AssertionError(msg)


_SUITE_DIRECTORIES: list[Path] = []
"""Every data directory the ``socket_for`` fixture has invented, so a path can be traced back."""


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
