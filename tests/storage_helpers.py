"""Fixtures and builders shared by the storage tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest_asyncio

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
from manicule.storage.engine import create_engine
from manicule.storage.migrator import upgrade

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest_asyncio.fixture
async def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest_asyncio.fixture
async def engine(data_dir: Path) -> AsyncIterator[AsyncEngine]:
    """A migrated database in a fresh data directory."""
    built = create_engine(data_dir)
    await upgrade(built)
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
