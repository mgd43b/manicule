"""What the Gmail connector suites share: a connector over the synthetic Gmail, and messages.

Everything runs the real client, pacer and connector against :class:`FakeGmail`; nothing here
touches the network or needs a Google account. Addresses are at ``example.org``.
"""

from __future__ import annotations

import zlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from email.message import EmailMessage
from email.policy import SMTP
from typing import Any

from pydantic import SecretStr

from manicule.connectors.config import GmailConfig
from manicule.connectors.gmail import GmailConnector
from manicule.connectors.gmail_client import GmailClient
from manicule.core.sources import DiscoveredDoc, Watermark
from tests.connectors.fake_gmail import (
    ADDRESS,
    FakeCredential,
    FakeGmail,
    FakeGmailSettings,
    SimulatedTime,
)

CLIENT_ID = "1234567890-example.apps.googleusercontent.com"


def gmail_config(**options: Any) -> GmailConfig:
    """A valid configuration with both secrets supplied, so no environment is read."""
    values: dict[str, object] = {
        "account": ADDRESS,
        "client_id": CLIENT_ID,
        "client_secret": SecretStr("client-secret-sentinel"),
        "refresh_token": SecretStr("refresh-token-sentinel"),
        **options,
    }
    return GmailConfig.model_validate(values)


def mail(
    subject: str,
    body: str = "The figures are in this sentence.",
    *,
    message_id: str | None = None,
    with_id: bool = True,
    attachment: bool = False,
) -> bytes:
    """A real message from the standard library's writer.

    It carries a ``Message-ID`` (derived from the subject unless one is given) because nearly
    every real message does and the link it yields is what citations show; ``with_id=False`` is
    the message that has none.
    """
    built = EmailMessage()
    built["From"] = "pat@example.org"
    built["To"] = ADDRESS
    built["Subject"] = subject
    built["Date"] = "Tue, 03 Feb 2026 09:14:00 +0000"
    if message_id is not None:
        built["Message-ID"] = message_id
    elif with_id:
        built["Message-ID"] = f"<{zlib.crc32(subject.encode()):08x}@example.org>"
    built.set_content(body + "\n")
    if attachment:
        built.add_attachment(
            b"Measured over the whole quarter rather than at the peak.\n",
            maintype="text",
            subtype="plain",
            filename="notes.txt",
        )
    return built.as_bytes(policy=SMTP)


def mailbox(*, page_size: int = 2, **options: Any) -> tuple[FakeGmail, SimulatedTime]:
    """A synthetic mailbox whose pacing and quota share one simulated clock.

    Pages of two by default, so every walk pages. A test about what a full page does asks for
    Gmail's own size instead.
    """
    time = SimulatedTime()
    return FakeGmail(clock=time, page_size=page_size, **options), time


def connector_over(
    server: FakeGmail,
    time: SimulatedTime,
    *,
    name: str = "mail",
    max_retries: int = 5,
    **options: Any,
) -> GmailConnector:
    """A connector, not yet set up, reading ``server`` with the real client."""
    settings = FakeGmailSettings(max_retries=max_retries)
    client = GmailClient(
        settings,
        FakeCredential(server),
        transport=server.transport(),
        sleep=time.sleep,
        clock=time,
    )
    return GmailConnector(
        gmail_config(**options),
        client,
        name=name,
        renewal="Run `manicule connector login mail`.",
        clock=time,
    )


@asynccontextmanager
async def opened(
    server: FakeGmail, time: SimulatedTime, **options: Any
) -> AsyncGenerator[GmailConnector]:
    """A connector that has been set up, torn down on the way out."""
    connector = connector_over(server, time, **options)
    await connector.setup()
    try:
        yield connector
    finally:
        await connector.teardown()


async def drain(
    connector: GmailConnector, watermark: Watermark | None = None
) -> list[DiscoveredDoc]:
    """Everything one ``discover`` yields, in order."""
    return [doc async for doc in connector.discover(watermark)]


def ids(documents: list[DiscoveredDoc]) -> list[str]:
    return [document.source_id for document in documents]
