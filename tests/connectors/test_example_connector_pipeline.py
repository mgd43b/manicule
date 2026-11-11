"""The example plugin's connector through the real pipeline, with durable acquisition on.

``assert_connector_contract`` checks the protocol in isolation and cannot see what the pipeline
asks of a connector on top of it. Durable acquisition (``storage.retain_source_bytes``, on by
default) refuses a fetched body that carries no proof of the revision it was read at, and leaves
every document of a connector that does not supply one in retry. A template that passed the
contract and indexed nothing is what this is here to prevent. See
``docs/writing-a-connector.md``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from manicule_plugin_example import DirectoryConnector, DirectoryConnectorConfig

from manicule.core.content import DocumentStatus
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import LanceVectorStore
from tests.connectors.gmail_support import mail_pipeline
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.ingest.pipeline import IngestPipeline

NAME = "notes"


@pytest_asyncio.fixture
async def vectors(data_dir: Path) -> AsyncIterator[LanceVectorStore]:
    store = LanceVectorStore(data_dir / "vectors")
    await store.ensure_ready(HashEmbedder().fingerprint)
    yield store
    await store.teardown()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    directory = tmp_path / "notes"
    directory.mkdir()
    (directory / "alpha.txt").write_text("The alpha note says hello.\n", encoding="utf-8")
    (directory / "beta.txt").write_text("The beta note says goodbye.\n", encoding="utf-8")
    return directory


@pytest.fixture
def pipeline(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, vectors: LanceVectorStore
) -> IngestPipeline:
    return mail_pipeline(store, engine, data_dir, vectors)


def connector(root: Path) -> DirectoryConnector:
    return DirectoryConnector(NAME, DirectoryConnectorConfig(root=root))


async def live(store: SqliteDocStore) -> dict[str, DocumentStatus]:
    documents = await store.list_documents(limit=1_000)
    return {d.source_id: d.status for d in documents if d.source == NAME}


@pytest.mark.parametrize("retain", [True, False])
async def test_every_note_is_indexed_with_and_without_retained_bytes(
    pipeline: IngestPipeline, store: SqliteDocStore, root: Path, retain: bool
) -> None:
    """``retain=True`` is the default, and the case a connector without revision proof fails."""
    report = await pipeline.run(connector(root), retain_source_bytes=retain)

    assert report.error == ""
    assert await live(store) == {
        "alpha.txt": DocumentStatus.INDEXED,
        "beta.txt": DocumentStatus.INDEXED,
    }
    document = await store.find_document(NAME, "alpha.txt")
    assert document is not None
    chunks = await store.document_chunks(document.id)
    assert "alpha note says hello" in "\n".join(chunk.text for chunk in chunks)


async def test_a_second_sync_changes_only_what_changed(
    pipeline: IngestPipeline, store: SqliteDocStore, root: Path
) -> None:
    await pipeline.run(connector(root))
    before = await store.find_document(NAME, "beta.txt")
    assert before is not None

    alpha = root / "alpha.txt"
    stat = alpha.stat()
    alpha.write_text("The alpha note now says something else.\n", encoding="utf-8")
    os.utime(alpha, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))

    report = await pipeline.run(connector(root))

    assert report.error == ""
    changed = await store.find_document(NAME, "alpha.txt")
    assert changed is not None
    text = "\n".join(chunk.text for chunk in await store.document_chunks(changed.id))
    assert "something else" in text
    unchanged = await store.find_document(NAME, "beta.txt")
    assert unchanged is not None
    assert unchanged.version_token == before.version_token
