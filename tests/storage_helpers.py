"""Fixtures and builders shared by the storage tests."""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text

from manicule.core.anchors import HeadingAnchor, Unlocated
from manicule.core.content import (
    NEEDS_ATTENTION,
    BlockKind,
    Chunk,
    Document,
    DocumentStatus,
    PipelineStage,
)
from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.storage.docstore import DEFAULT_WORKSPACE, SqliteDocStore
from manicule.storage.engine import create_engine, database_path, prepare_data_dir
from manicule.storage.migrator import upgrade

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest_asyncio.fixture
async def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


async def _build_template(directory: Path) -> None:
    """Migrate one database to head and leave nothing in its write-ahead log.

    The checkpoint is what makes the file copyable: SQLite keeps committed pages in the
    ``-wal`` until one, so copying the database alone before it would hand back a schema
    missing whatever the last revisions wrote.
    """
    built = create_engine(directory)
    try:
        await upgrade(built)
        async with built.connect() as connection:
            await connection.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
    finally:
        await built.dispose()


@pytest.fixture(scope="session")
def migrated_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One database at head, built once, for :func:`engine` to copy.

    **Replaying 33 revisions per test was 34% of this suite.** 834 tests take a migrated
    database and every one of them produced the same schema the same way, at 0.42s against the
    0.004s a file copy costs — 120x, and the whole of it setup rather than anything a test
    asserts. Alembic still runs; it runs once.

    Nothing is given up. A test gets its own file, containing the same tables and the same
    ``alembic_version`` as before, so isolation and the schema under test are unchanged. The
    migration *path* is not what this fixture ever exercised — ``test_storage_migrations.py``
    upgrades and downgrades every revision explicitly, and the suites that need a database at
    some older revision call ``upgrade(engine, revision=...)`` themselves and are untouched.

    Session-scoped, which under ``pytest-xdist`` means once per worker: four builds of 0.46s
    against the 349s of replay they replace. Synchronous on purpose — ``asyncio_default_fixture
    _loop_scope`` is ``function``, so a session-scoped *async* fixture would outlive the loop it
    was made on; ``asyncio.run`` here owns a loop that ends with the build.
    """
    directory = tmp_path_factory.mktemp("migrated-template")
    asyncio.run(_build_template(directory))
    return database_path(directory)


@pytest_asyncio.fixture
async def engine(data_dir: Path, migrated_template: Path) -> AsyncIterator[AsyncEngine]:
    """A migrated database in a fresh data directory.

    Copied from :func:`migrated_template` rather than migrated in place. ``prepare_data_dir``
    runs first so the directory is the ``0700`` the product makes rather than whatever the
    umask says, and ``copy2`` carries the database's own mode across with it.
    """
    prepare_data_dir(data_dir)
    shutil.copy2(migrated_template, database_path(data_dir))
    built = create_engine(data_dir)
    try:
        yield built
    finally:
        await built.dispose()


@pytest_asyncio.fixture
async def store(engine: AsyncEngine) -> SqliteDocStore:
    made = SqliteDocStore(engine)
    await made.ensure_workspace()
    return made


def make_document(
    source: str = "fs",
    source_id: str = "s1",
    *,
    workspace_id: str = DEFAULT_WORKSPACE,
    status: DocumentStatus = DocumentStatus.INDEXED,
    uri: str = "file:///a.md",
    title: str = "A",
    media_type: str = "text/markdown",
    body: bytes = b"hello",
) -> Document:
    """A document with the invariants the model insists on already satisfied."""
    detail = "synthetic detail" if status in NEEDS_ATTENTION else None
    return Document(
        id=document_id(workspace_id, source, source_id),
        source=source,
        source_id=source_id,
        uri=uri,
        title=title,
        content_hash=content_hash(body),
        media_type=media_type,
        status=status,
        status_detail=detail,
        failed_stage=PipelineStage.PARSE if status is DocumentStatus.FAILED else None,
    )


def make_chunk(
    document: Document,
    position: int,
    text: str,
    *,
    heading_path: tuple[str, ...] = ("Auth", "Tokens"),
    kind: BlockKind = BlockKind.PROSE,
    located: bool = True,
    lang: str | None = None,
) -> Chunk:
    """A chunk whose id is derived exactly as ingest would derive it."""
    breadcrumb = " > ".join(heading_path)
    embed_text = f"{breadcrumb}\n\n{text}" if breadcrumb else text
    anchor = (
        HeadingAnchor(path=heading_path, fragment=None)
        if located
        else Unlocated(reason="synthetic chunk with no location")
    )
    return Chunk(
        id=chunk_id(document.id, position, text),
        document_id=document.id,
        text=text,
        embed_text=embed_text,
        anchor=anchor,
        heading_path=heading_path,
        kind=kind,
        position=position,
        token_count=max(1, len(text.split())),
        metadata={"lang": lang} if lang is not None else {},
    )


def fingerprint(dimension: int = 8, model_id: str = "test/model") -> EmbedFingerprint:
    return EmbedFingerprint(
        model_id=model_id,
        dimension=dimension,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=512,
    )


__all__ = [
    "data_dir",
    "engine",
    "fingerprint",
    "make_chunk",
    "make_document",
    "store",
]
