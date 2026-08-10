"""Organisation on top of the corpus: collections, tags, versions, relations, the trash.

The corpus itself is documents and chunks. This module is the vocabulary for everything a
person imposes on it afterwards — grouping documents, labelling them, tracking what they used
to say, linking chunks to one another, and taking a document out of circulation without
destroying it.

Two rules from elsewhere shape every type here and are worth stating once.

**Identity is workspace-scoped** (``docs/storage.md`` §4.2). A collection, a tag and a
document all belong to exactly one workspace, and nothing in this module carries a workspace
id — the store handle does. That is deliberate: a value that named its own workspace could
disagree with the handle that fetched it, and the disagreement would be a tenancy bug wearing
a data type. :class:`CollectionRule` is the sharp case, because it is *stored* and later
re-executed, and a stored rule that could name a workspace would be a saved query capable of
widening its own scope.

**A location is correct, or it is absent** (``docs/contracts.md`` §1). Applied to versioning,
that is :class:`CitationResolution`: a citation into a superseded version does not resolve to
"the nearest thing", it resolves to *absent*, and it says which kind of absent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.content import Chunk, Document, Metadata


class _Organisation(BaseModel):
    """Organisation types are frozen values, like the content types they describe."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- collections ---------------------------------------------------------------------------


class CollectionRule(_Organisation):
    """A stored restriction that decides membership without anyone listing documents.

    A rule-driven collection is a saved query. Membership is then the union of the documents
    somebody added by hand and the documents the rule currently selects, evaluated at read
    time — so "everything from the runbooks space" keeps meaning that as the corpus grows,
    rather than meaning "what was there the day it was created".

    **It carries no workspace, and it never will.** A rule is stored, and it is re-executed
    later by whichever handle reads the collection. If the rule could name a workspace, a saved
    query would be able to widen its own scope past the handle evaluating it, which is the
    exact shape of a cross-tenant leak: nothing raises, results arrive, and they are somebody
    else's documents. The evaluating store supplies the workspace, always.

    **An empty rule is refused.** A rule that restricts nothing selects the whole workspace,
    and a collection that silently contains every document is indistinguishable from a
    collection somebody meant to fill in. Leaving :attr:`Collection.rule` unset is how "no
    rule" is spelled; there is deliberately not a second spelling that means the opposite of
    what it looks like.
    """

    sources: frozenset[str] = frozenset()
    """Connector instance names, matched against ``documents.source``."""

    media_types: frozenset[str] = frozenset()
    tag_ids: frozenset[str] = frozenset()
    """Tags a document must carry. Membership in *any* of them, matching the disjunction-
    within-a-field convention :class:`~manicule.core.retrieval.Filter` uses."""

    updated_after: datetime | None = None
    updated_before: datetime | None = None

    @model_validator(mode="after")
    def _restricts_something(self) -> Self:
        if not self.restricting_fields:
            msg = (
                "a CollectionRule must restrict something. A rule with no fields set selects "
                "every document in the workspace, which is never what anyone meant by adding a "
                "rule; leave Collection.rule unset instead"
            )
            raise ValueError(msg)
        for name in ("updated_after", "updated_before"):
            value: datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware; a naive timestamp has no defined meaning"
                raise ValueError(msg)
        return self

    @property
    def restricting_fields(self) -> frozenset[str]:
        """The fields this rule actually restricts on."""
        return frozenset(
            name
            for name, field in type(self).model_fields.items()
            if getattr(self, name) != field.get_default(call_default_factory=True)
        )


class Collection(_Organisation):
    """A named set of documents, filled by hand, by a rule, or by both."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    rule: CollectionRule | None = None
    created_at: datetime


# --- tags ----------------------------------------------------------------------------------


class Tag(_Organisation):
    """An arbitrary label, unique by name within a workspace."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str | None = None


# --- versions ------------------------------------------------------------------------------


class DocumentVersion(_Organisation):
    """A state a document has **left**, recorded at the moment it was superseded.

    The state a document is in *now* lives in ``documents`` and has no row here. That
    asymmetry is not an oversight: a version row records what the document was, and every
    field of it — the content hash, the retained bytes, the chunk count — is only complete
    once the state is finished with. Recording the incoming state instead would write a row
    whose ``original_ref`` is filled in a moment later, and a history whose most recent entry
    is the one that might still be wrong.

    So :attr:`version` counts supersessions: version 1 is the first state the document left,
    and the state it holds now is one past the highest row.
    """

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    content_hash: str = Field(min_length=1)

    original_ref: str | None = Field(
        default=None,
        description="The bytes this version was built from, while they are still retained. "
        "``None`` means they were never kept, or that the retention window has passed and "
        "``release_expired_versions`` has let the blob store reclaim them.",
    )
    chunk_count: int | None = Field(
        default=None, ge=0, description="How many chunks this version had when it was replaced."
    )
    changes: Metadata = Field(
        default_factory=dict,
        description="Which fields differed between this version and the one that replaced it. "
        "Diagnostic, not a patch: it says what moved, never how to move it back.",
    )
    superseded_at: datetime


class CitationState(StrEnum):
    """What became of the text a citation named.

    The point of naming four outcomes rather than returning a chunk or ``None`` is that three
    of them are absences with different remedies, and an operator holding a citation that no
    longer resolves needs to know which one they have.
    """

    PRESENT = "present"
    """The chunk is stored and its document is not in the trash. The citation resolves.

    **Servability is a separate question and is deliberately not folded in here.** A document
    mid-re-index is not ``indexed`` and must not appear in a search — that boundary belongs to
    the hydrating join, and :func:`manicule.testing.assert_pipeline_enforces_scope` holds it
    there. But the text this citation named is in the store, and answering "absent" about it
    would be wrong in the direction that matters: it would report a citation as broken while
    the passage it quotes is sitting in the row.
    """

    SUPERSEDED = "superseded"
    """The document was re-ingested and no longer contains that text.

    ``chunks.id`` is derived from ``(document_id, position, text)``, so a chunk that survived
    a re-parse unchanged kept its id — and one whose text or position moved did not. The old
    id therefore *dangles* rather than silently re-pointing at different text, which is the
    behaviour ``docs/storage.md`` §3.2 chose it for. Nothing in manicule resolves this to the
    superseding text: the citation quoted a passage that is gone, and offering the paragraph
    that replaced it as though it were the same passage is precisely the substitution the
    anchor rules exist to forbid.
    """

    DELETED = "deleted"
    """The document is in the trash, or its content was purged after the grace period.

    Restorable — freely inside the grace period, and by a re-parse from retained bytes
    outside it (``docs/ingest.md`` §11.2).
    """

    UNKNOWN = "unknown"
    """No document of that id in this workspace, and therefore nothing to say about the chunk.

    Distinct from :attr:`SUPERSEDED` on purpose. "The text changed" is an answer; "this store
    has never heard of that document" is a different one, and reporting the first when the
    second is true would send somebody looking through a version history that does not exist.
    """


class CitationResolution(_Organisation):
    """The answer to "does this citation still point at something", with its reason."""

    state: CitationState
    chunk: Chunk | None = None
    document: Document | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _present_means_present(self) -> Self:
        if (self.state is CitationState.PRESENT) != (self.chunk is not None):
            msg = (
                "a CitationResolution carries a chunk exactly when its state is 'present'; got "
                f"state={self.state.value!r} chunk={'set' if self.chunk else 'unset'}"
            )
            raise ValueError(msg)
        return self

    @property
    def resolved(self) -> bool:
        return self.state is CitationState.PRESENT


# --- the trash -----------------------------------------------------------------------------


class TrashEntry(_Organisation):
    """One soft-deleted document, and how much of it is left.

    :attr:`purged` is the difference between a restore that costs nothing and a restore that
    costs a re-parse. Inside the grace period a soft-deleted document keeps its chunks, its
    vectors and its lexical rows — all invisible at the hydrating join — so clearing the
    timestamp puts it straight back into service. After the sweep has been through, the
    content is gone and the row is a headstone.
    """

    document: Document
    deleted_at: datetime
    purged: bool = False
    restorable_until: datetime | None = Field(
        default=None,
        description="When the sweep becomes entitled to purge this document's content. "
        "``None`` once it already has.",
    )

    @property
    def free_restore(self) -> bool:
        """Whether restoring costs nothing but clearing a timestamp."""
        return not self.purged


class Restoration(_Organisation):
    """What restoring a document actually achieved, and what is still needed.

    A restore that returns nothing leaves the caller unable to tell "back in service" from
    "the row exists again and holds no text". Those need different follow-ups, and only one of
    them is finished.
    """

    document_id: str = Field(min_length=1)
    restored: bool
    needs_reparse: bool = False
    """Whether the document is back but empty, so its content has to be re-derived.

    It says *that* the content is missing, not which rung of the blast-radius ladder gets it
    back. With retained bytes the repair is a single-document re-parse, rung 3, on this
    machine; without them the only remedy is a forced re-sync from the source, rung 4, which
    can fail for reasons outside the machine. :attr:`reason` says which of the two applies, and
    ``Document.original_ref`` is what decides it.
    """

    reason: str = Field(min_length=1)


# --- chunk relations -----------------------------------------------------------------------


class ChunkRelationType(StrEnum):
    """The typed edges chunks may carry.

    Closed here rather than in the database. ``chunk_relations.relation_type`` is plain
    ``TEXT`` with no ``CHECK``, and that is the right side of the trade the schema conventions
    describe (``docs/storage.md`` §3.4): a misspelled ``documents.status`` makes a document
    invisible to retrieval forever and silently, which is why *that* value set is enforced by a
    constraint. A misspelled relation type produces an edge no query asks for — visible, inert
    and reversible — while a database constraint would make a plugin-defined relation a schema
    migration. The vocabulary is pinned where the meaning is, and a store validates against it
    on the way in.
    """

    PARENT = "parent"
    """``source`` is a child of ``target``. Read the row as "source's parent is target"."""

    SIBLING = "sibling"
    """The two chunks are peers. Symmetric, and therefore stored **once**.

    One row, not two. ``docs/storage.md`` §4.4 keeps an index on ``target_chunk_id`` precisely
    because lookups are ``WHERE source = ? OR target = ?``, and a composite key leading with
    ``source`` cannot serve the second half of that predicate. Writing the mirror row as well
    would double the table to serve a query the schema is already indexed for, and would
    introduce a pair that can fall out of step.
    """


class ChunkEdge(_Organisation):
    """One typed link between two chunks, as stored.

    Direction is a property of the row, not of the query, so a lookup from either end returns
    the edge as it was written. A caller asking "what is above this chunk" reads
    :meth:`points_away_from`; one that only wants the neighbour reads :meth:`other_than`.
    """

    source_chunk_id: str = Field(min_length=1)
    target_chunk_id: str = Field(min_length=1)
    relation_type: ChunkRelationType

    @model_validator(mode="after")
    def _not_reflexive(self) -> Self:
        if self.source_chunk_id == self.target_chunk_id:
            msg = (
                f"a chunk cannot relate to itself; got {self.source_chunk_id!r} on both ends. "
                f"The schema refuses this too, so building one here only moves the failure."
            )
            raise ValueError(msg)
        return self

    def other_than(self, chunk_id: str) -> str:
        """The chunk at the far end of this edge from ``chunk_id``.

        Raises:
            ValueError: ``chunk_id`` is at neither end, which means the edge came from
                somewhere other than a lookup on it.
        """
        if chunk_id == self.source_chunk_id:
            return self.target_chunk_id
        if chunk_id == self.target_chunk_id:
            return self.source_chunk_id
        msg = (
            f"chunk {chunk_id!r} is at neither end of the edge "
            f"{self.source_chunk_id!r} -> {self.target_chunk_id!r}"
        )
        raise ValueError(msg)

    def points_away_from(self, chunk_id: str) -> bool:
        """Whether ``chunk_id`` is this edge's source.

        For :attr:`ChunkRelationType.PARENT` that is the difference between "the parent of this
        chunk" and "a child of this chunk", which no amount of set membership recovers.
        """
        return chunk_id == self.source_chunk_id


__all__ = [
    "ChunkEdge",
    "ChunkRelationType",
    "CitationResolution",
    "CitationState",
    "Collection",
    "CollectionRule",
    "DocumentVersion",
    "Restoration",
    "Tag",
    "TrashEntry",
]
