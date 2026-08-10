"""Wiring for the connector suites: a configured connector over a synthetic instance.

Every helper here builds the **real** client against an injected transport rather than a mock
of the client, so the retry loop, the header construction and — above all — the query encoding
are the ones that would run against a live instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from pydantic import SecretStr

from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import ConfluenceConfig, Deployment
from manicule.connectors.confluence import ConfluenceConnector
from manicule.core.sources import DiscoveredDoc
from manicule.testing import closing
from tests.connectors.fake_confluence import CLOUD_BASE, FakeConfluence

__all__ = ["Waits", "cloud_config", "connected", "drain", "ids", "server_config"]


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


def client_for(
    instance: FakeConfluence,
    config: ConfluenceConfig | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[ConfluenceConfig, ConfluenceClient]:
    settings = config if config is not None else cloud_config(base_url=instance.base_url)
    return settings, ConfluenceClient(
        settings,
        transport=instance.transport(),
        sleep=sleep if sleep is not None else Waits(),
        clock=clock if clock is not None else (lambda: 0.0),
    )


async def connected(
    instance: FakeConfluence,
    config: ConfluenceConfig | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> ConfluenceConnector:
    """A connector wired to ``instance`` and already set up. The caller tears it down."""
    settings, client = client_for(instance, config, sleep=sleep, clock=clock)
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
