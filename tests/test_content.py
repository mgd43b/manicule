"""Content types, and the statuses that keep failures visible."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manicule.core.anchors import LineAnchor
from manicule.core.content import (
    CHUNKLESS_BY_DESIGN,
    IN_FLIGHT,
    NEEDS_ATTENTION,
    SETTLED,
    Chunk,
    Document,
    DocumentStatus,
    PipelineStage,
    RawDocument,
)
from manicule.core.ids import chunk_id, content_hash, document_id
from tests.fakes import MEDIA_TYPE, make_document


def _document(**overrides: object) -> Document:
    base: dict[str, object] = {
        "id": document_id("w", "s", "1"),
        "source": "s",
        "source_id": "1",
        "uri": "s://1",
        "content_hash": content_hash("x"),
        "media_type": MEDIA_TYPE,
    }
    return Document.model_validate({**base, **overrides})


def test_a_pdf_with_no_text_gets_a_status_of_its_own() -> None:
    """Optical character recognition is out of scope; a scanned PDF must still be visible.

    An empty document indexed as a success is the failure this status exists to prevent.
    """
    document = _document(
        status=DocumentStatus.NO_EXTRACTABLE_TEXT,
        status_detail="no text layer; the pages are images",
    )
    assert document.needs_attention
    assert not document.expects_chunks
    assert document.status is not DocumentStatus.INDEXED


def test_getting_nothing_is_not_the_same_as_crashing() -> None:
    """Different remedies: one wants OCR, the other wants a bug fixed."""
    empty = _document(status=DocumentStatus.NO_EXTRACTABLE_TEXT, status_detail="image-only pages")
    crashed = _document(
        status=DocumentStatus.FAILED,
        status_detail="xref table is corrupt",
        failed_stage=PipelineStage.PARSE,
    )
    assert empty.status is not crashed.status
    assert empty.failed_stage is None
    assert crashed.failed_stage is PipelineStage.PARSE


def test_an_archive_is_not_a_document_that_failed() -> None:
    """Its members became documents of their own; zero chunks is the correct outcome."""
    archive = _document(status=DocumentStatus.CONTAINER)
    assert not archive.needs_attention
    assert not archive.expects_chunks


@pytest.mark.parametrize("status", sorted(NEEDS_ATTENTION))
def test_every_bad_outcome_must_explain_itself(status: DocumentStatus) -> None:
    """An unexplained failure is not actionable, so the type refuses to hold one."""
    with pytest.raises(ValidationError, match="status_detail"):
        _document(
            status=status,
            failed_stage=PipelineStage.PARSE if status is DocumentStatus.FAILED else None,
        )


def test_failed_documents_name_the_stage_and_others_do_not() -> None:
    with pytest.raises(ValidationError, match="failed_stage"):
        _document(status=DocumentStatus.FAILED, status_detail="boom")
    with pytest.raises(ValidationError, match="failed_stage"):
        _document(status=DocumentStatus.INDEXED, failed_stage=PipelineStage.EMBED)


def test_only_a_document_on_its_way_to_being_indexed_may_carry_chunks() -> None:
    """Every other status stores zero chunks and zero vectors.

    The in-flight states belong here rather than in ``CHUNKLESS_BY_DESIGN`` because chunks are
    written *before* vectors and before the commit: a document in ``embedding`` legitimately
    has chunks and is not yet servable. Being chunkless "by design" is a statement about a
    settled document, not about one that is mid-flight.
    """
    carrying = {status for status in DocumentStatus if status not in CHUNKLESS_BY_DESIGN}
    assert carrying == {
        DocumentStatus.PENDING,
        DocumentStatus.FETCHING,
        DocumentStatus.PARSING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.PARSED,
        DocumentStatus.INDEXED,
    }


def test_every_status_is_either_settled_or_on_its_way_somewhere() -> None:
    """No status may be missing from both sets, or a document in it is swept by nothing.

    The recovery sweep reads :data:`IN_FLIGHT` and change detection reads :data:`SETTLED`.
    A member in neither is a document that is never requeued *and* never skipped — quietly
    re-ingested from scratch on every run, forever.
    """
    accounted = IN_FLIGHT | SETTLED | {DocumentStatus.PENDING, DocumentStatus.PARSED}
    assert accounted | {DocumentStatus.DELETED} == set(DocumentStatus)


def test_raw_documents_decode_either_way() -> None:
    as_text = RawDocument(source_id="1", uri="u", media_type="text/plain", content="héllo")
    as_bytes = RawDocument(
        source_id="1", uri="u", media_type="text/plain", content="héllo".encode()
    )
    assert as_text.as_bytes() == as_bytes.as_bytes()
    assert as_text.as_text() == as_bytes.as_text() == "héllo"


def test_chunks_carry_no_score() -> None:
    """A score belongs to a retrieval run. Storing one invites code that reads a stale number."""
    assert "score" not in Chunk.model_fields


def test_embed_text_is_separate_from_the_text_that_gets_cited() -> None:
    """A section called "Configuration" is unretrievable without knowing what it configures.

    The breadcrumb has to reach the embedder and must not reach the quotation.
    """
    document = make_document()
    chunk = Chunk(
        id=chunk_id(document.id, 0, "the timeout is 30s"),
        document_id=document.id,
        text="the timeout is 30s",
        embed_text="Guide > Configuration > the timeout is 30s",
        anchor=LineAnchor(start=1, end=1),
        heading_path=("Guide", "Configuration"),
        position=0,
        token_count=5,
    )
    assert chunk.text == "the timeout is 30s"
    assert chunk.text in chunk.embed_text
    assert "Configuration" not in chunk.text


def test_pipeline_stages_are_in_ingest_order() -> None:
    """The six stages, in order — and ``middleware`` after them, because it is not one.

    Hooks run *between* stages. A hook that raises is a plugin problem, and recording it under
    the stage it happened to bound would send an operator to read a parser that worked
    perfectly, so it gets a value of its own. Placing it last is what keeps the six in order.
    """
    assert list(PipelineStage) == [
        PipelineStage.DISCOVER,
        PipelineStage.FETCH,
        PipelineStage.PARSE,
        PipelineStage.CHUNK,
        PipelineStage.EMBED,
        PipelineStage.STORE,
        PipelineStage.MIDDLEWARE,
    ]
