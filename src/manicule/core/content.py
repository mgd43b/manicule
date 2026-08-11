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
    """The ingest stages, in order, plus the boundary between them.

    One vocabulary serves three purposes: attributing a failure to a stage, labelling
    metrics, and naming middleware hook points. Three enums that had to agree would
    eventually stop agreeing.

    :attr:`MIDDLEWARE` is last and is deliberately not a stage. Hooks run *between* stages,
    and a hook that raises is a plugin problem rather than a problem with the stage it
    happened to bound — filing it under ``parse`` would send an operator to read a parser
    that worked perfectly. It is a value of ``failed_stage`` so that "everything a hook
    broke" stays a query.
    """

    DISCOVER = "discover"
    FETCH = "fetch"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    STORE = "store"
    MIDDLEWARE = "middleware"


class DocumentStatus(StrEnum):
    """Where a document got to, and why it stopped if it stopped.

    Every state a document can end an ingest run in is nameable. A document that produced
    nothing must never look the same as one that indexed cleanly.
    """

    PENDING = "pending"
    """Discovered, or requeued by the recovery sweep; no stage has claimed it yet."""

    FETCHING = "fetching"
    """The source bytes are being fetched, in the pipeline's own process."""

    PARSING = "parsing"
    """The parser chain is running, in a subprocess that may be killed."""

    EMBEDDING = "embedding"
    """Chunks are written and vectors are being produced."""

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


IN_FLIGHT: frozenset[DocumentStatus] = frozenset(
    {
        DocumentStatus.FETCHING,
        DocumentStatus.PARSING,
        DocumentStatus.EMBEDDING,
    }
)
"""The statuses a document holds only while a process is working on it.

**An allowlist, never a denylist, and that is the whole point of naming it here.** The
recovery sweep requeues documents stuck in these states after a crash, and the tempting
formulation — everything that is not ``indexed`` — swallows :attr:`DocumentStatus.CONTAINER`
and :attr:`DocumentStatus.NO_EXTRACTABLE_TEXT`, both of which are terminal and both of which
have zero chunks by design. Requeued forever, they would be re-fetched, re-parsed and
re-requeued on every run.

An allowlist fails closed: a status added later is simply not swept until somebody adds it
here. A denylist fails open, and the failure is silent.
"""

SETTLED: frozenset[DocumentStatus] = frozenset(
    {
        DocumentStatus.INDEXED,
        DocumentStatus.CONTAINER,
        DocumentStatus.NO_EXTRACTABLE_TEXT,
        DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
        DocumentStatus.FAILED,
        DocumentStatus.SKIPPED,
    }
)
"""Statuses in which a stored document is a finished answer about the bytes it holds.

Change detection may skip a document only when it is one of these, and the reason is a bug
that is otherwise very hard to see. A crash mid-ingest leaves a document requeued to
``pending`` — but with its ``version_token`` and ``content_hash`` already written. A skip
rule that consulted only the token would then skip it forever, and a document that never
finished indexing would be re-skipped on every sync while appearing, to every count, to have
been handled.

Another allowlist, for the same reason as :data:`IN_FLIGHT`: the members are named, so a
status added later is not silently treated as finished.
"""

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

    @property
    def lang(self) -> str | None:
        """The language this chunk records, or ``None`` when it is undetermined.

        Read through :attr:`metadata` rather than stored as a field of its own, because
        ``docs/contracts.md`` §2 fixes what a chunk *is* and this is a chunker-supplied
        annotation rather than part of that identity. The chunker sets it only when a chunk's
        blocks agree on a language.

        It is a property rather than two `metadata.get` calls because both stores promote it
        into a column that :attr:`~manicule.core.retrieval.Filter.langs` resolves against —
        ``chunks.lang`` in SQLite and ``lang`` in the Lance table — and a promotion rule that
        differed between them would return two different corpora for one filter. ``None``
        means undetermined, which is not the same as any particular language and never matches
        an ``IN`` list.
        """
        value = self.metadata.get("lang")
        return value if isinstance(value, str) else None


class Retention(_Content):
    """What became of a document's source bytes when the pipeline offered them for keeping.

    Exactly one of the two is set. ``ref`` present means re-parsing this document never means
    re-fetching it — rung 3 of the blast-radius ladder, and the only thing standing between a
    parser bug fix and a full re-crawl of a rate-limited API. ``omitted_reason`` present means
    the bytes were not kept and says why, on the same principle as
    :class:`~manicule.core.anchors.Unlocated`: absent with a stated reason, visible in
    diagnostics, never a silent partial success.

    Lives in core because it is the vocabulary two sides speak — the store that retains and
    the pipeline that records the outcome — and neither should have to import the other.
    """

    ref: str | None = None
    omitted_reason: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.ref is None) == (self.omitted_reason is None):
            msg = (
                "a Retention carries either a ref or a reason it has none; "
                f"got ref={self.ref!r} omitted_reason={self.omitted_reason!r}"
            )
            raise ValueError(msg)
        return self


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

    parse_fp: str | None = Field(
        default=None,
        description="Canonical "
        ":class:`~manicule.core.fingerprints.ParseFingerprint` of the parser run that "
        "produced this document's stored text and anchors, or ``None`` when there is no "
        "recorded lineage. **Read here, written only through "
        ":meth:`~manicule.ingest.ports.IngestStore.set_lineage`** — like ``chunk_fp`` and "
        "``embed_fp``, which for that reason are not on this model at all. This one is, "
        "because change detection reads it: a document whose bytes have not moved but whose "
        "parser has is not unchanged, and deciding that needs the stored value in hand. A "
        "store that wrote lineage from here would clear it on every ingest, since the "
        "pipeline builds a fresh document per run and cannot know a fingerprint before the "
        "chain has chosen a parser.",
    )

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
    "IN_FLIGHT",
    "NEEDS_ATTENTION",
    "SETTLED",
    "BlockKind",
    "Chunk",
    "Document",
    "DocumentStatus",
    "Metadata",
    "ParsedBlock",
    "PipelineStage",
    "RawDocument",
    "Retention",
]
