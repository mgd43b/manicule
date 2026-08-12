"""Wiring for the connector suites: a configured connector over a synthetic instance.

Every helper here builds the **real** client against an injected transport rather than a mock
of the client, so the retry loop, the header construction and — above all — the query encoding
are the ones that would run against a live instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import AuthMethod, ConfluenceConfig, Deployment
from manicule.connectors.confluence import ConfluenceConnector
from manicule.connectors.credentials import BrowserSession, BrowserSessionCredential, Credential
from manicule.core.sources import DiscoveredDoc
from manicule.testing import closing
from tests.connectors.fake_confluence import CLOUD_BASE, FakeConfluence

__all__ = [
    "CAPTURED_AT",
    "SESSION_ACCOUNT",
    "Waits",
    "browser_session",
    "cloud_config",
    "connected",
    "drain",
    "ids",
    "server_config",
    "sso_config",
]

CAPTURED_AT = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
"""When every test session was captured. Fixed so the age check is a matter of arithmetic."""

SESSION_ACCOUNT = "sync.user"
"""Who the fake instance says a captured session belongs to."""


class Waits:
    """A stand-in for sleeping, which records what it was asked to wait for.

    Injected so that a test of the backoff asserts on the delay rather than serving it: a
    suite that actually waits out a ``Retry-After`` is a suite nobody runs.
    """

    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


class Clock:
    """A monotonic clock a test moves by hand, for the cursor-lifetime check."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cloud_config(**overrides: object) -> ConfluenceConfig:
    settings: dict[str, object] = {
        "base_url": CLOUD_BASE,
        "deployment": Deployment.CLOUD,
        "email": "sync@example.com",
        "api_token": SecretStr("token"),
        "page_size": 2,
    }
    settings.update(overrides)
    return ConfluenceConfig.model_validate(settings)


def server_config(base_url: str, **overrides: object) -> ConfluenceConfig:
    settings: dict[str, object] = {
        "base_url": base_url,
        "deployment": Deployment.SERVER,
        "personal_access_token": SecretStr("pat"),
        "page_size": 2,
    }
    settings.update(overrides)
    return ConfluenceConfig.model_validate(settings)


def sso_config(base_url: str, **overrides: object) -> ConfluenceConfig:
    """A Server/Data Center instance whose identity provider left it with no tokens to offer."""
    settings: dict[str, object] = {
        "base_url": base_url,
        "deployment": Deployment.SERVER,
        "auth": AuthMethod.BROWSER_SESSION,
        "page_size": 2,
    }
    settings.update(overrides)
    return ConfluenceConfig.model_validate(settings)


def browser_session(
    config: ConfluenceConfig,
    *,
    account: str = SESSION_ACCOUNT,
    captured_at: datetime | None = None,
    cookies: dict[str, str] | None = None,
    now: datetime | None = None,
) -> BrowserSessionCredential:
    """A captured session, as ``manicule connector login`` would have stored one."""
    when = captured_at if captured_at is not None else CAPTURED_AT
    session = BrowserSession(
        base_url=config.base_url,
        account=account,
        captured_at=when,
        cookies={
            name: SecretStr(value)
            for name, value in (
                cookies or {"JSESSIONID": "ABC123", "seraph.confluence": "77"}
            ).items()
        },
    )
    moment = now if now is not None else when
    return BrowserSessionCredential(
        session=session,
        max_age=timedelta(hours=config.session_max_age_hours),
        now=lambda: moment,
    )


def client_for(
    instance: FakeConfluence,
    config: ConfluenceConfig | None = None,
    *,
    credential: Credential | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[ConfluenceConfig, ConfluenceClient]:
    settings = config if config is not None else cloud_config(base_url=instance.base_url)
    return settings, ConfluenceClient(
        settings,
        credential=credential,
        transport=instance.transport(),
        sleep=sleep if sleep is not None else Waits(),
        clock=clock if clock is not None else (lambda: 0.0),
    )


async def connected(
    instance: FakeConfluence,
    config: ConfluenceConfig | None = None,
    *,
    credential: Credential | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> ConfluenceConnector:
    """A connector wired to ``instance`` and already set up. The caller tears it down."""
    settings, client = client_for(instance, config, credential=credential, sleep=sleep, clock=clock)
    connector = ConfluenceConnector(settings, client)
    await connector.setup()
    return connector


async def drain[T](iterator: AsyncIterator[T]) -> list[T]:
    """Consume an async iterator to completion, closing it afterwards."""
    async with closing(iterator) as stream:
        return [item async for item in stream]


def ids(documents: Sequence[DiscoveredDoc]) -> list[str]:
    """Source ids of discovered documents, for readable assertions."""
    return [document.source_id for document in documents]
