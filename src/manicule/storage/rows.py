"""Conversion between ORM rows and the frozen domain types.

One place, because several surfaces need it — documents are returned by listing, by a
collection, by a tag and by the trash — and a second copy of the mapping is the field nobody
remembers when the type gains one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter

from manicule.core.anchors import Anchor
from manicule.core.content import LEGACY_PUBLICATION, BlockKind, Chunk, Document, DocumentStatus
from manicule.core.ids import vector_id
from manicule.storage import models
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from manicule.core.content import Metadata

_ANCHOR: TypeAdapter[Anchor] = TypeAdapter(Anchor)


def to_document(row: models.Document) -> Document:
    return Document(
        id=row.id,
        publication_id=row.publication_id,
        source=row.source,
        source_id=row.source_id,
        uri=row.uri,
        title=row.title,
        content_hash=row.content_hash,
        version_token=row.version_token,
        original_ref=row.original_ref,
        media_type=row.media_type,
        status=row.status,
        status_detail=row.status_detail,
        failed_stage=row.failed_stage,
        metadata=cast("Metadata", row.doc_metadata or {}),
        parse_fp=row.parse_fp,
        # Read out, never written back — `apply_document` below stamps the column itself, on the
        # same rule as `parse_fp`. A citation reports it, and of the three timestamps a mirrored
        # document makes it easy to confuse it is the only one that is a fact about *this*
        # installation rather than about the source or about whoever took the snapshot.
        indexed_at=row.indexed_at,
    )


def apply_document(row: models.Document, document: Document) -> None:
    """Write a domain document onto a row, clearing any soft delete.

    Writing a document is asserting that it exists, so an upsert clears a soft delete. A
    document removed at the source and later restored there arrives through exactly this path,
    and leaving the timestamp would index it into a row nothing can see.

    **Lineage is not written here.** ``parse_fp``, ``chunk_fp`` and ``embed_fp`` move only
    through :meth:`~manicule.storage.docstore.DocumentStore.set_lineage`, which is called
    after a stage commits. The pipeline builds a fresh :class:`Document` for every ingest and
    cannot know a parse fingerprint before the chain has chosen a parser, so writing lineage
    from the domain object would clear it at the start of every run and leave "which
    documents need re-parsing" answering "all of them" for the wrong reason.
    """
    row.deleted_at = None
    row.publication_id = document.publication_id
    row.source = document.source
    row.source_id = document.source_id
    row.uri = document.uri
    row.title = document.title
    row.media_type = document.media_type
    row.content_hash = document.content_hash
    row.version_token = document.version_token
    row.original_ref = document.original_ref
    row.status = document.status
    row.status_detail = document.status_detail
    row.failed_stage = document.failed_stage
    row.doc_metadata = cast("Any", dict(document.metadata))
    if document.status is DocumentStatus.INDEXED:
        row.indexed_at = utcnow()


def chunk_values(
    chunk: Chunk, document_id: str, publication_id: str = LEGACY_PUBLICATION
) -> dict[str, Any]:
    """One chunk as column values, for a bulk insert.

    The same mapping :func:`from_chunk` builds an ORM instance from, kept as the single
    definition of what a chunk row *is* so the two cannot describe different columns.

    It exists because the ORM cannot batch this insert. ``Chunk.seq`` is a server-generated
    ``INTEGER PRIMARY KEY``, so the unit of work needs the generated keys back in parameter
    order and asks for ``RETURNING``; the SQLite dialect declares no insertmanyvalues sentinel,
    so it falls back to one statement per row. Measured: fifty ``session.add`` calls produce
    fifty ``INSERT`` statements, each carrying ``RETURNING``, where one Core ``insert()`` over
    the same rows produces one.
    """
    return {
        "id": chunk.id,
        "vector_id": vector_id(publication_id, chunk.id),
        "document_id": document_id,
        "text": chunk.text,
        "embed_text": chunk.embed_text,
        "heading_text": " > ".join(chunk.heading_path),
        "heading_path": list(chunk.heading_path),
        "kind": chunk.kind,
        # Promoted into a column so the lexical leg can filter on it, through the one
        # accessor the Lance row also uses (`docs/retrieval.md` §3.3).
        "lang": chunk.lang,
        "position": chunk.position,
        "token_count": chunk.token_count,
        "anchor": chunk.anchor.model_dump(mode="json"),
        "chunk_metadata": cast("Any", dict(chunk.metadata)),
    }


def from_chunk(
    chunk: Chunk, document_id: str, publication_id: str = LEGACY_PUBLICATION
) -> models.Chunk:
    """One chunk as an ORM instance, for a caller that needs the mapped object."""
    return models.Chunk(**chunk_values(chunk, document_id, publication_id))


def to_chunk(row: models.Chunk) -> Chunk:
    heading_path = cast("list[str]", row.heading_path or [])
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        text=row.text,
        embed_text=row.embed_text,
        anchor=_ANCHOR.validate_python(row.anchor),
        heading_path=tuple(heading_path),
        kind=BlockKind(row.kind),
        position=row.position,
        token_count=row.token_count,
        metadata=cast("Metadata", row.chunk_metadata or {}),
    )


__all__ = ["apply_document", "chunk_values", "from_chunk", "to_chunk", "to_document"]
