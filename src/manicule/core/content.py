"""The content types that flow through ingest: raw bytes to blocks to chunks.

One idea shapes all of them: **structure is discovered once, by the component that can see
it, and never re-derived downstream from prose.** A parser can see that something is a
table because it is reading the table markup; a chunker looking at the extracted string
cannot, and any attempt to recover it there is guesswork.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from manicule.core.anchors import Anchor

Metadata = dict[str, JsonValue]
"""Free-form per-item metadata. JSON-shaped so it survives the round trip to storage."""


class BlockKind(StrEnum):
    """What a parsed block is.

    This is what lets a chunker keep a table or a code block whole instead of severing it
    at a character count. A parser that labels everything ``prose`` has thrown away the
    only signal the chunker has.
    """

    PROSE = "prose"
    HEADING = "heading"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    PANEL = "panel"
    MEDIA = "media"


class PipelineStage(StrEnum):
    """The ingest stages, in order.

    One vocabulary serves three purposes: attributing a failure to a stage, labelling
    metrics, and naming middleware hook points. Three enums that had to agree would
    eventually stop agreeing.
    """

    DISCOVER = "discover"
    FETCH = "fetch"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    STORE = "store"


class DocumentStatus(StrEnum):
    """Where a document got to, and why it stopped if it stopped.

    Every state a document can end an ingest run in is nameable. A document that produced
    nothing must never look the same as one that indexed cleanly.
    """

    PENDING = "pending"
    """Discovered or fetched; not yet parsed."""

    PARSED = "parsed"
    """Parsed into at least one block; not yet embedded and stored.

    A visible intermediate state, so a run interrupted between parse and embed resumes
    from where it stopped instead of re-parsing.
    """

    INDEXED = "indexed"
    """Parsed, chunked, embedded and stored. The success state for a document with text."""

    CONTAINER = "container"
    """An archive whose members were expanded into documents of their own.

    Zero chunks by design. Without a name of its own, an archive is indistinguishable from
    a document that yielded nothing, and gets reported as a fault it does not have.
    """

    NO_EXTRACTABLE_TEXT = "no_extractable_text"
    """The parser chain ran to completion without error and produced no text.

    The usual cause is a scanned or image-only PDF. Optical character recognition is out of
    scope, and this status is how that stays visible: such a document is reported, not
    quietly indexed as empty and returned to nobody. It is also the selector for the
    re-parse pass that runs when that decision is revisited.

    Distinct from :attr:`FAILED` at :attr:`PipelineStage.PARSE`, and the distinction is the
    point — "we got nothing" and "we crashed" want different remedies.
    """

    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    """No installed parser claimed the media type, and no fallback did either.

    Stored rather than dropped, so what was skipped is visible.
    """

    FAILED = "failed"
    """An error stopped this document. ``failed_stage`` says where, ``status_detail`` why.

    One failed document never aborts a batch; the batch records it and continues.
    """

    SKIPPED = "skipped"
    """Excluded before parsing by configuration or by middleware."""

    DELETED = "deleted"
    """Removed at the source, or by a user. Retained so citations can explain themselves."""


NEEDS_ATTENTION: frozenset[DocumentStatus] = frozenset(
    {
        DocumentStatus.NO_EXTRACTABLE_TEXT,
        DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
        DocumentStatus.FAILED,
    }
)
"""Statuses a diagnostic command must surface. Silence about these is the failure mode."""

CHUNKLESS_BY_DESIGN: frozenset[DocumentStatus] = frozenset(
    {
        DocumentStatus.CONTAINER,
        DocumentStatus.NO_EXTRACTABLE_TEXT,
        DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
        DocumentStatus.SKIPPED,
        DocumentStatus.FAILED,
        DocumentStatus.DELETED,
    }
)
"""Statuses that store zero chunks and zero vectors.

A placeholder empty chunk, added so that every document "has" one, would be retrievable
and would cite nothing. These documents have no chunks, and that is the honest record.
"""


class _Content(BaseModel):
    """Content types are frozen: they are values that get stored, not mutable state."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RawDocument(_Content):
    """Source bytes as a connector delivered them, before any interpretation."""

    source_id: str = Field(min_length=1, description="Stable identifier within its source.")
    uri: str = Field(min_length=1, description="Where a human would go to see this.")
    media_type: str = Field(min_length=1, description="IANA media type, e.g. ``application/pdf``.")
    content: bytes | str
    encoding: str = "utf-8"
    metadata: Metadata = Field(default_factory=dict)

    def as_bytes(self) -> bytes:
        """The content as bytes, encoding text with :attr:`encoding` if needed."""
        if isinstance(self.content, bytes):
            return self.content
        return self.content.encode(self.encoding)

    def as_text(self) -> str:
        """The content as text, decoding bytes with :attr:`encoding` if needed."""
        if isinstance(self.content, str):
            return self.content
        return self.content.decode(self.encoding)


class ParsedBlock(_Content):
    """One structural unit of a document, with the location it came from.

    Blocks are emitted in reading order. Every block carries an anchor — an
    :class:`~manicule.core.anchors.Unlocated` one when the parser genuinely cannot place it,
    never a guess.
    """

    kind: BlockKind
    text: str
    anchor: Anchor
    heading_path: tuple[str, ...] = ()
    lang: str | None = Field(
        default=None,
        description="Language tag: a code language for ``code`` blocks, a natural language "
        "elsewhere. ``None`` means undetermined, not 'English'.",
    )
    metadata: Metadata = Field(default_factory=dict)


class Chunk(_Content):
    """A retrievable unit of text with everything needed to cite it.

    There is deliberately no score here. A score belongs to a retrieval run, not to stored
    content: the same chunk scores differently for every query, and storing one invites
    code that reads a stale number.
    """

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)

    text: str
    """What gets shown and cited. Exactly the source text, with nothing added."""

    embed_text: str
    """What the embedder sees.

    Carries the heading breadcrumb prefixed to :attr:`text`, because a section titled
    "Configuration" is unretrievable without knowing what it configures. Keeping the two
    apart means retrieval scaffolding never leaks into a quotation.
    """

    anchor: Anchor
    heading_path: tuple[str, ...] = ()
    kind: BlockKind = BlockKind.PROSE
    position: int = Field(ge=0, description="0-based ordinal within its document.")
    token_count: int = Field(ge=0)
    metadata: Metadata = Field(default_factory=dict)


class Document(_Content):
    """The indexed record of one source document.

    Chunks and vectors hang off this; a citation resolves through it.
    """

    id: str = Field(min_length=1)

    source: str = Field(min_length=1, description="Name of the connector instance that owns it.")
    source_id: str = Field(
        min_length=1,
        description="Stable identifier within that source. Reconciliation compares these, "
        "so a document that cannot supply one cannot be deletion-checked.",
    )
    uri: str = Field(min_length=1)
    title: str = ""

    content_hash: str = Field(min_length=1, description="Hash of the retained source bytes.")

    version_token: str | None = Field(
        default=None,
        description="Opaque and connector-defined: a git blob SHA, a Confluence "
        "``version.number``, an S3 ETag. Compared for equality and never interpreted.",
    )
    original_ref: str | None = Field(
        default=None,
        description="Pointer to the retained source bytes, so re-parsing never means re-fetching.",
    )

    media_type: str = Field(min_length=1)
    status: DocumentStatus = DocumentStatus.PENDING
    status_detail: str | None = Field(
        default=None,
        description="Why the status is what it is. Required for every status in "
        "``NEEDS_ATTENTION``, because an unexplained failure is not actionable.",
    )
    failed_stage: PipelineStage | None = Field(
        default=None,
        description="Which stage raised. Set exactly when the status is ``failed``, so "
        "'re-run everything that died in parse' is a query rather than a grep.",
    )
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _failures_are_explained(self) -> Self:
        if self.status in NEEDS_ATTENTION and not self.status_detail:
            msg = f"status {self.status.value!r} requires a status_detail explaining it"
            raise ValueError(msg)
        if (self.status is DocumentStatus.FAILED) != (self.failed_stage is not None):
            msg = (
                "failed_stage must be set for status 'failed' and unset otherwise; "
                f"got status={self.status.value!r} failed_stage={self.failed_stage!r}"
            )
            raise ValueError(msg)
        return self

    @property
    def needs_attention(self) -> bool:
        """True when this document did not index and somebody should be told."""
        return self.status in NEEDS_ATTENTION

    @property
    def expects_chunks(self) -> bool:
        """True when this document should have chunks stored against it."""
        return self.status not in CHUNKLESS_BY_DESIGN


__all__ = [
    "CHUNKLESS_BY_DESIGN",
    "NEEDS_ATTENTION",
    "BlockKind",
    "Chunk",
    "Document",
    "DocumentStatus",
    "Metadata",
    "ParsedBlock",
    "PipelineStage",
    "RawDocument",
]
