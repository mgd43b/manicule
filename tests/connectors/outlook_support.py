"""What the Outlook connector suites share: a connector over the synthetic Graph, and messages.

Everything runs the real client, retry loop and connector against :class:`FakeGraph`; nothing here
touches the network or needs a Microsoft 365 tenant. Addresses are at ``example.org``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from manicule.connectors.outlook import OutlookConnector
from manicule.connectors.outlook_client import OutlookClient
from manicule.connectors.outlook_config import OutlookConfig
from manicule.core.sources import DiscoveredDoc, Watermark
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.workers import InProcessRunner
from manicule.parsers.calendar import CalendarParser
from manicule.parsers.config import CalendarConfig, MailConfig
from manicule.parsers.mail import MailParser
from manicule.storage.blobs import BlobStore
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import LanceVectorStore
from tests.connectors.fake_gmail import SimulatedTime
from tests.connectors.fake_graph import (
    ADDRESS,
    CLIENT_ID,
    TENANT,
    FakeCredential,
    FakeGraph,
    FakeGraphSettings,
)
from tests.fakes import HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
"""The wall clock every connector built here reads, so the calendar window is the same every day."""


def outlook_config(**options: Any) -> OutlookConfig:
    """A valid configuration with a refresh token supplied, so no environment is read."""
    values: dict[str, object] = {
        "account": ADDRESS,
        "tenant_id": TENANT,
        "client_id": CLIENT_ID,
        "refresh_token": SecretStr("refresh-token-sentinel-7730"),
        **options,
    }
    return OutlookConfig.model_validate(values)


def tenant() -> tuple[FakeGraph, SimulatedTime]:
    """A synthetic tenant in pages of two, so every walk pages, and its simulated clock."""
    return FakeGraph(page_size=2), SimulatedTime()


def connector_over(
    server: FakeGraph,
    time: SimulatedTime,
    *,
    name: str = "work",
    max_retries: int = 5,
    **options: Any,
) -> OutlookConnector:
    """A connector, not yet set up, reading ``server`` with the real client."""
    client = OutlookClient(
        FakeGraphSettings(max_retries=max_retries),
        FakeCredential(server),
        transport=server.transport(),
        sleep=time.sleep,
    )
    return OutlookConnector(
        outlook_config(**options),
        client,
        name=name,
        renewal="Run `manicule connector login work`.",
        clock=time,
        now=lambda: NOW,
    )


def _chain(media_type: str) -> list[str]:
    """Which parsers read what, as the real chain does by media type."""
    return ["calendar", "lines"] if media_type == "text/calendar" else ["email", "lines"]


def pipeline_for(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    vectors: LanceVectorStore,
    **options: Any,
) -> IngestPipeline:
    """The real pipeline, with the mail and calendar parsers, durable acquisition and bytes."""
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=store,
        acquisitions=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner(
            {
                "email": MailParser(MailConfig()),
                "calendar": CalendarParser(CalendarConfig()),
                "lines": fakes.LineParser(),
            }
        ),
        resolve_chain=_chain,
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        blobs=BlobStore(engine, data_dir),
        **options,
    )


@asynccontextmanager
async def opened(
    server: FakeGraph, time: SimulatedTime, **options: Any
) -> AsyncGenerator[OutlookConnector]:
    """A connector that has been set up, torn down on the way out."""
    connector = connector_over(server, time, **options)
    await connector.setup()
    try:
        yield connector
    finally:
        await connector.teardown()


async def drain(
    connector: OutlookConnector, watermark: Watermark | None = None
) -> list[DiscoveredDoc]:
    """Everything one ``discover`` yields, in order."""
    return [doc async for doc in connector.discover(watermark)]


def ids(documents: list[DiscoveredDoc]) -> list[str]:
    return [document.ref.source_id for document in documents]
