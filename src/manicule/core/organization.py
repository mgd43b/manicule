"""Organization on top of the corpus: collections, tags, versions, relations, the trash.

The corpus itself is documents and chunks. This module is the vocabulary for everything a
person imposes on it afterwards — grouping documents, labeling them, tracking what they used
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

import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from manicule.core.content import Chunk, Document, Metadata


class _Organization(BaseModel):
    """Organization types are frozen values, like the content types they describe."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- collections ---------------------------------------------------------------------------


_DRIVE_LETTER = 1
"""The longest a URI scheme may not be, because a Windows drive letter is exactly this long."""


def directory_prefix(value: str) -> str:
    """One entry of :attr:`CollectionRule.uri_prefixes`, in the form the clause compares.

    Two normalizations, and each exists because the alternative is a rule that matches nothing
    and says nothing about why.

    **An absolute path becomes the URI the connector recorded.** ``documents.uri`` holds
    ``Path.as_uri()`` for a filesystem source, so a rule spelled ``/corpus/journals`` would
    never meet a row. Converting here means the recommended spelling is the one a person
    already knows, and it is :meth:`~pathlib.Path.as_uri` doing the percent-encoding rather
    than a hand-written ``file://`` string, which gets it wrong on the first directory with a
    space in its name.

    **A prefix gains a trailing ``/``.** Membership is a comparison against text, so a
    boundary that is not in the text is not in the comparison: without it ``…/journals`` also
    selects ``…/journals-old``, silently and in the widening direction. With it, the stored
    form is a literal prefix of exactly the URIs beneath that directory.

    Raises:
        ValueError: The value is neither an absolute path nor a URI carrying a scheme.
            Relative to nothing is not a location — a rule is stored and re-run later, by a
            process whose working directory is nobody's business.
    """
    text = value.strip()
    if not text:
        msg = (
            "a collection rule uri prefix must be an absolute path or a URI, not an empty "
            "string. Name the directory its documents sit in, such as '/corpus/journals'"
        )
        raise ValueError(msg)
    if Path(text).is_absolute():
        # Lexical, never `Path.resolve`: the rule may name a corpus that lives on the machine
        # that will evaluate it rather than the one writing it, and resolving would consult
        # *this* filesystem for symlinks that are not its business.
        text = Path(os.path.normpath(text)).as_uri()
    # Longer than a drive letter, because `urlsplit` reads `C:/corpus` as the scheme `c`, and
    # storing that would be a prefix no connector can ever write.
    elif len(urlsplit(text).scheme) <= _DRIVE_LETTER:
        msg = (
            f"collection rule uri prefix {value!r} is neither an absolute path nor a URI with "
            f"a scheme. A prefix is matched against the location a connector recorded, so it "
            f"has to name one: '/corpus/journals' for a local tree, or "
            f"'https://wiki.example/spaces/RUN' for a site"
        )
        raise ValueError(msg)
    return text if text.endswith("/") else f"{text}/"


class CollectionRule(_Organization):
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

    uri_prefixes: frozenset[str] = frozenset()
    """Directory subtrees, matched against ``documents.uri`` as a literal prefix.

    **This is where "the collection is the directory" gets said once.** ``authoring.collections``
    already holds that a collection name is also the directory beneath the root its documents
    land in, but only :meth:`~manicule.app.service.ApplicationService.document_create` acted on
    it, by writing a membership row per document. A file that arrived any other way — a git
    pull, an editor, a ``connector sync`` over a tree somebody else filled — got none, and a
    collection-scoped search returned less with nothing to say why. A prefix is that same
    sentence in the one place membership is decided.

    **It matches the address, not the identity.** ``documents.source_id`` is what a source
    promises is stable and ``documents.uri`` is where the document sits; a rule about location
    belongs on the location. Two consequences follow, both wanted. A document that moves
    between directories changes collection, at read time and without re-indexing, which is what
    a directory-shaped collection should do and what materializing membership would get wrong.
    And a document whose sidecar manifest declares a canonical address is matched on *that*
    address rather than on the copy's path — so a mirror is selected by the space it mirrors,
    which is the distinction :mod:`~manicule.connectors.filesystem` draws between a mirror and
    a directory, kept rather than contradicted.

    Entries are normalized by :func:`directory_prefix`: an absolute path becomes a ``file:``
    URI and every entry ends in ``/``. Membership in *any* of them, like :attr:`tag_ids`."""

    media_types: frozenset[str] = frozenset()
    tag_ids: frozenset[str] = frozenset()
    """Tags a document must carry. Membership in *any* of them, matching the disjunction-
    within-a-field convention :class:`~manicule.core.retrieval.Filter` uses."""

    updated_after: datetime | None = None
    updated_before: datetime | None = None

    @field_validator("sources", "media_types", "tag_ids")
    @classmethod
    def _selectors_are_not_empty(cls, values: frozenset[str]) -> frozenset[str]:
        if any(not value.strip() for value in values):
            msg = "collection rule selectors must be non-empty strings"
            raise ValueError(msg)
        return values

    @field_validator("uri_prefixes")
    @classmethod
    def _prefixes_name_a_location(cls, values: frozenset[str]) -> frozenset[str]:
        """Normalized on the way in, so the stored rule is the form the clause compares.

        Here rather than in :func:`~manicule.storage.organization.rule_clause`, because a rule
        is public collection metadata that round-trips through four surfaces: normalizing at
        evaluation would show an operator a prefix that is not the one being matched.
        """
        return frozenset(directory_prefix(value) for value in values)

    @field_serializer("sources", "media_types", "tag_ids", "uri_prefixes", when_used="json")
    def _sorted_selectors(self, values: frozenset[str]) -> list[str]:
        """Give every public surface one deterministic representation of set-valued fields."""
        return sorted(values)

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


class Collection(_Organization):
    """A named set of documents, filled by hand, by a rule, or by both."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    rule: CollectionRule | None = None
    created_at: datetime


# --- tags ----------------------------------------------------------------------------------


class Tag(_Organization):
    """An arbitrary label, unique by name within a workspace."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str | None = None


# --- versions ------------------------------------------------------------------------------


class DocumentVersion(_Organization):
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
    behavior ``docs/storage.md`` §3.2 chose it for. Nothing in manicule resolves this to the
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


class CitationResolution(_Organization):
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


class TrashEntry(_Organization):
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


class Restoration(_Organization):
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

    LINKS_TO = "links_to"
    """``source`` declares a link to ``target``. Directional, and asymmetric in meaning.

    What a *declared* reference looks like in a corpus of markdown: a link that is the whole of
    a list item, with or without a leading verb — ``- [[other]]``, ``relates_to [[other]]``.
    Somebody writing that is stating a relationship rather than mentioning a name in passing,
    and the inverse ("other is linked to from here") is a different fact, so the row is written
    once in the direction it was written in and read from both ends.
    """

    MENTIONS = "mentions"
    """``source`` refers to ``target`` in prose. A soft reference.

    **Separate from** :attr:`LINKS_TO` **rather than folded into it**, even though a corpus is
    usually mostly this. The two are not the same claim: a passing mention inside a sentence is
    evidence that two documents are about related things, while a declared link is an assertion
    by the author about how they relate. Collapsing them would make the distinction
    unrecoverable — and it is the distinction a reader would want first when asking what a
    document is actually connected to, since ranking by mention count and ranking by declared
    links give different answers on the same corpus.
    """


class ChunkEdge(_Organization):
    """One typed link between two chunks, as stored.

    Direction is a property of the row, not of the query, so a lookup from either end returns
    the edge as it was written. A caller asking "what is above this chunk" reads
    :meth:`points_away_from`; one that only wants the neighbor reads :meth:`other_than`.
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
    "directory_prefix",
]
