"""``manicule connector login`` for Outlook: the authorization code flow, end to end.

Everything runs for real except Microsoft's two hosts: the loopback listener is a socket on this
machine, the browser's visit is an HTTP request to it, and the token endpoint and ``GET /me`` are
answered by :class:`~tests.connectors.fake_graph.FakeGraph` through the mock transport. What is
asserted is what an operator and a reviewer care about: the request Microsoft is sent, what is
refused, where the token ends up, and that it ends up nowhere else.
"""

from __future__ import annotations

import io
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import pytest

from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors.errors import CredentialError, MailboxMismatchError
from manicule.connectors.outlook_login import (
    FileSink,
    StreamSink,
    granted_scopes_are_acceptable,
)
from manicule.core.errors import ConfigError, PolicyError
from tests.app.fakes import FakeBackend
from tests.connectors.fake_google_signin import visit
from tests.connectors.fake_graph import ADDRESS, CLIENT_ID, TENANT, FakeGraph

G = "https://graph.microsoft.com/"
WANTED = ("offline_access", "User.Read", "Mail.Read", "Calendars.Read")
OPTIONS = {"account": ADDRESS, "tenant_id": TENANT, "client_id": CLIENT_ID}


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTLOOK_REFRESH_TOKEN", raising=False)


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def service() -> ApplicationService:
    backend = FakeBackend()
    backend.settings = Settings(
        connectors={
            "work": ConnectorSettings(type="outlook", options=dict(OPTIONS)),
            "notes": ConnectorSettings(type="filesystem", options={"root": "/srv/notes"}),
            "off": ConnectorSettings(type="outlook", enabled=False, options=dict(OPTIONS)),
            "broken": ConnectorSettings(type="outlook", options={"account": ADDRESS}),
            "loop": ConnectorSettings(
                type="outlook", options={**OPTIONS, "login_redirect_host": "127.0.0.1"}
            ),
        }
    )
    return ApplicationService(backend)


def settled(visits: list[int], count: int = 1, *, timeout: float = 10.0) -> list[int]:
    """The visits, once ``count`` are recorded.

    The browser stub runs on the command's own browser thread, and the command returns, or raises,
    as soon as its socket has answered, while that thread is still reading the answer; asserting
    on the list directly races the thread appending to it (it did, on CI).
    """
    deadline = time.monotonic() + timeout
    while len(visits) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return list(visits)


def browser(graph: FakeGraph, visits: list[int], **choices: Any) -> Callable[[str], bool]:
    """A browser that sends the person's choice to the loopback socket the command opened."""

    def open_it(url: str) -> bool:
        visits.append(visit(graph.consent(url, **choices)))
        return True

    return open_it


async def login(
    service: ApplicationService,
    graph: FakeGraph,
    sink: Any,
    *,
    name: str = "work",
    open_browser: Callable[[str], object] | None = None,
    timeout_seconds: float = 20.0,
    visits: list[int] | None = None,
) -> Any:
    return await service.connector_login_outlook(
        name,
        sink=sink,
        timeout_seconds=timeout_seconds,
        open_browser=open_browser or browser(graph, visits if visits is not None else []),
        transport=graph.transport(),
    )


async def test_the_whole_flow_writes_the_minted_token_to_a_private_file(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    target = tmp_path / "outlook-token"
    visits: list[int] = []
    issued = await login(service, graph, FileSink(target), visits=visits)

    written = target.read_text().strip()
    assert written.startswith("refresh-token-rotated-"), "the token Entra issued for the code"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert issued.name == "work"
    assert issued.account == ADDRESS
    assert set(issued.scope.split()) == {
        "offline_access",
        "User.Read",
        "Mail.Read",
        "Calendars.Read",
    }
    assert written not in issued.model_dump_json(), "the result never carries the token"
    assert settled(visits) == [200]


async def test_the_request_microsoft_is_sent_is_a_public_client_code_flow_with_pkce(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    await login(service, graph, FileSink(tmp_path / "t"))

    asked = graph.authorizations[0]
    assert asked["client_id"] == CLIENT_ID
    assert asked["login_hint"] == ADDRESS
    assert asked["redirect_uri"].startswith("http://localhost:")
    assert asked["code_challenge_method"] == "S256"
    assert "client_secret" not in asked
    exchange = next(r for r in graph.requests if b"authorization_code" in r.content)
    assert b"client_secret" not in exchange.content
    assert b"code_verifier=" in exchange.content


async def test_the_redirect_host_is_the_one_the_instance_registered(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    await login(service, graph, FileSink(tmp_path / "t"), name="loop")

    assert graph.authorizations[0]["redirect_uri"].startswith("http://127.0.0.1:")


async def test_a_declined_sign_in_issues_nothing_and_writes_nothing(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    target = tmp_path / "t"
    with pytest.raises(CredentialError, match="declined"):
        await login(
            service, graph, FileSink(target), open_browser=browser(graph, [], approve=False)
        )

    assert not target.exists()
    assert graph.exchanges == 0


async def test_a_forged_redirect_is_refused_before_its_code_is_looked_at(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    target = tmp_path / "t"
    visits: list[int] = []
    with pytest.raises(CredentialError, match="state"):
        await login(
            service,
            graph,
            FileSink(target),
            open_browser=browser(graph, visits, state="forged"),
        )

    assert not target.exists()
    assert graph.exchanges == 0
    assert settled(visits) == [400]


async def test_a_token_for_another_account_is_discarded_and_neither_address_is_printed(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    graph.upn = "someone.else@contoso.example.org"
    graph.mail_address = "someone.else@example.org"
    target = tmp_path / "t"
    with pytest.raises(MailboxMismatchError) as refused:
        await login(service, graph, FileSink(target))

    assert not target.exists(), "a token for the wrong mailbox is written nowhere"
    assert "someone.else" not in str(refused.value)
    assert ADDRESS not in str(refused.value)
    assert "circular" not in str(refused.value)
    assert "connector login" not in str(refused.value), "login is not the cure for a login"


@pytest.mark.parametrize(
    "granted",
    [
        f"{G}Mail.ReadWrite {G}Calendars.Read User.Read",
        f"{G}Mail.Read {G}Mail.Send User.Read",
        f"{G}Mail.Read User.Read",  # Calendars.Read missing
    ],
)
async def test_a_token_that_carries_other_scopes_than_the_ones_asked_for_is_discarded(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path, granted: str
) -> None:
    graph.granted_scope = granted
    target = tmp_path / "t"
    with pytest.raises(CredentialError, match="read-only scopes"):
        await login(service, graph, FileSink(target))

    assert not target.exists()


async def test_the_destination_is_checked_before_the_browser_opens(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    existing = tmp_path / "t"
    existing.write_text("precious")
    opened: list[str] = []

    with pytest.raises(ConfigError, match="already exists"):
        await login(
            service,
            graph,
            FileSink(existing),
            open_browser=opened.append,
        )

    assert opened == []
    assert existing.read_text() == "precious"


async def test_a_terminal_is_refused_as_a_destination(
    service: ApplicationService, graph: FakeGraph
) -> None:
    class Terminal(io.StringIO):
        @override
        def isatty(self) -> bool:
            return True

    with pytest.raises(ConfigError, match="terminal"):
        await login(service, graph, StreamSink(Terminal()))


async def test_a_pipe_receives_the_token_and_nothing_else(
    service: ApplicationService, graph: FakeGraph
) -> None:
    pipe = io.StringIO()
    await login(service, graph, StreamSink(pipe))

    assert pipe.getvalue().startswith("refresh-token-rotated-")
    assert pipe.getvalue().count("\n") == 1


async def test_a_code_entra_will_not_redeem_is_reported_by_its_code_and_nothing_else(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    def lose_the_code(url: str) -> bool:
        redirect = graph.consent(url)
        graph.exchange_failure = (400, "invalid_client")
        visit(redirect)
        return True

    with pytest.raises(CredentialError, match="public client flows") as refused:
        await login(service, graph, FileSink(tmp_path / "t"), open_browser=lose_the_code)

    assert "sentinel-prose" not in str(refused.value)


async def test_no_sign_in_within_the_timeout_issues_nothing(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    with pytest.raises(CredentialError, match="no sign-in arrived"):
        await login(
            service,
            graph,
            FileSink(tmp_path / "t"),
            open_browser=lambda _url: True,
            timeout_seconds=0.3,
        )


async def test_only_an_enabled_outlook_source_has_a_token_to_mint(
    service: ApplicationService, graph: FakeGraph, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="nothing else has one"):
        await login(service, graph, FileSink(tmp_path / "t"), name="notes")
    with pytest.raises(PolicyError, match="enabled = false"):
        await login(service, graph, FileSink(tmp_path / "t"), name="off")
    with pytest.raises(ConfigError, match="not valid"):
        await login(service, graph, FileSink(tmp_path / "t"), name="broken")


# --- what a granted scope may look like ----------------------------------------------------------


@pytest.mark.parametrize(
    "granted",
    [
        f"{G}Mail.Read {G}Calendars.Read User.Read",
        "mail.read calendars.read user.read",
        "openid profile email offline_access Mail.Read Calendars.Read User.Read",
        "https%3A%2F%2Fgraph.microsoft.com%2FMail.Read Calendars.Read User.Read",
    ],
)
def test_the_scope_shapes_entra_really_sends_are_accepted(granted: str) -> None:
    assert granted_scopes_are_acceptable(granted, WANTED)


@pytest.mark.parametrize(
    "granted",
    [
        "Mail.ReadWrite Calendars.Read User.Read",
        "Mail.Read Calendars.Read User.Read Files.Read",
        "Mail.Read User.Read",
        "",
    ],
)
def test_anything_wider_or_narrower_than_asked_is_not(granted: str) -> None:
    assert not granted_scopes_are_acceptable(granted, WANTED)
