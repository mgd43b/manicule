"""Components that break their half of the bargain, so the guards can be seen firing.

Every fake here is written to be *wrong* in one specific, named way. That is the point: a
guard checked only against a component that behaves correctly is a guard nobody has watched
work, and this project has been bitten by exactly that.

The one to read carefully is :class:`LeakyStore`, which ignores its workspace scope entirely.
It exists so that ``tests/app/test_tenancy.py`` cannot be satisfied by a service that happens
not to be asked a cross-tenant question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, override

from manicule.app.ports import (
    Answering,
    Conversing,
    DocumentSurface,
    Ingesting,
    Keys,
    Maintenance,
    Organising,
    Retrieving,
    Telemetry,
)
from manicule.app.results import ApiKeySummary, Check
from manicule.app.tenancy import belongs_to
from manicule.config.settings import Settings
from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus
from manicule.core.embedding import IndexFingerprints
from manicule.core.errors import NameInUseError, UnknownEntityError
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.organisation import Collection as DocumentCollection
from manicule.core.organisation import CollectionRule, Restoration, Tag, TrashEntry
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.retrieval import Candidate, Confidence, ConfidenceBand, Context, Query
from manicule.generation.answers import AnswerEnvelope, AnswerEvent, EventKind
from manicule.generation.history import Turn
from manicule.generation.ports import (
    ConversationRecord,
    Feedback,
    FeedbackReason,
    SharedTurn,
)
from manicule.generation.sharing import ShareLink, redact_for_anonymous
from manicule.ingest.pipeline import RunReport
from manicule.ingest.reindex import ReindexReport
from manicule.retrieval.retriever import RetrievalResult
from manicule.storage.organisation import normalise_name

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Mapping, Sequence

    from manicule.core.retrieval import Filter
    from manicule.generation.answering import AnswerRequest, AnswerResult
    from manicule.plugins.registry import Discovery


def make_document(
    workspace: str,
    *,
    source: str = "local",
    source_id: str = "notes.md",
    title: str = "Notes",
    status: DocumentStatus = DocumentStatus.INDEXED,
    provenance: Provenance | None = None,
    indexed_at: datetime | None = None,
) -> Document:
    """A document whose id is derived the way the real one is.

    Derived rather than written down, so a test cannot be made to pass by editing a literal
    id until it matches. The identity under test *is* this function's second line.

    ``provenance`` goes into ``metadata`` under the reserved key rather than onto a field of its
    own, because that is where the real pipeline puts it and where every reader looks — a fixture
    that supplied it any other way would be exercising a path production does not have.
    """
    return Document(
        id=document_id(workspace, source, source_id),
        source=source,
        source_id=source_id,
        uri=f"file:///corpus/{source_id}",
        title=title,
        content_hash=content_hash(f"{workspace}/{source_id}"),
        media_type="text/markdown",
        status=status,
        metadata={PROVENANCE_KEY: provenance.as_metadata_value()} if provenance else {},
        indexed_at=indexed_at,
    )


def make_chunk(document: Document, *, text: str = "the client retries twice") -> Chunk:
    """One chunk of a document, with a resolvable anchor."""
    return Chunk(
        id=chunk_id(document.id, 0, text),
        document_id=document.id,
        text=text,
        embed_text=text,
        anchor=HeadingAnchor(path=("Retry policy",)),
        heading_path=("Retry policy",),
        kind=BlockKind.PROSE,
        position=0,
        token_count=6,
    )


@dataclass
class FakeStore:
    """A well-behaved document store, scoped to one workspace.

    The control against which :class:`LeakyStore` is the experiment: everything a test asserts
    about a refusal has to be shown *not* to happen here, or the test is only proving that the
    service refuses everything.
    """

    workspace_id: str = "default"
    documents: dict[str, Document] = field(default_factory=dict[str, Document])
    chunks: dict[str, list[Chunk]] = field(default_factory=dict[str, list[Chunk]])
    deleted: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def add(self, document: Document, *chunks: Chunk) -> Document:
        self.documents[document.id] = document
        self.chunks[document.id] = list(chunks)
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        """A page, filtered the way a correct store filters it.

        Honouring ``document_ids`` and the workspace scope matters here rather than being
        pedantry about a fake: a correct store is the control the leaky one is measured
        against, and a control that also leaked would make the experiment meaningless.
        """
        wanted = list(self.documents.values())
        if filter is not None:
            wanted = [
                document
                for document in wanted
                if belongs_to(self.workspace_id, document)
                and (not filter.document_ids or document.id in filter.document_ids)
                and (not filter.sources or document.source in filter.sources)
                and (not filter.media_types or document.media_type in filter.media_types)
            ]
        return wanted[offset : offset + limit]

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]:
        return self.chunks.get(document_id, [])

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
    ) -> int:
        chosen = [
            document
            for document in self.documents.values()
            if (source is None or document.source == source)
            and (statuses is None or document.status in statuses)
        ]
        return len(chosen)

    async def count_chunks(self, document_id: str | None = None) -> int:
        if document_id is not None:
            return len(self.chunks.get(document_id, []))
        return sum(len(chunks) for chunks in self.chunks.values())

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append((document_id, "hard"))
        self.documents.pop(document_id, None)

    async def soft_delete_document(self, document_id: str) -> None:
        self.deleted.append((document_id, "soft"))

    async def document_statistics(self) -> Mapping[str, Mapping[str, int]]:
        by_source: dict[str, int] = {}
        by_media: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for document in self.documents.values():
            by_source[document.source] = by_source.get(document.source, 0) + 1
            by_media[document.media_type] = by_media.get(document.media_type, 0) + 1
            by_status[document.status.value] = by_status.get(document.status.value, 0) + 1
        return {"by_source": by_source, "by_media_type": by_media, "by_status": by_status}

    async def index_fingerprints(self) -> IndexFingerprints:
        return IndexFingerprints()

    async def connector_metadata(self, connector: str) -> Mapping[str, object]:
        del connector
        return {}


class LeakyStore(FakeStore):
    """A store that ignores its workspace scope. **Deliberately broken.**

    Two ways at once, because a surface guard has to survive both:

    * ``get_document`` returns any document it holds, whatever workspace minted its id —
      inherited, because :class:`FakeStore` does not scope that lookup either.
    * ``list_documents`` returns **every** document it holds, ignoring the filter, the limit
      and the offset.

    Ignoring the *limit* as well as the filter is what makes this useful. A leaky store that
    still truncated would let the surface's "some of what I asked for came back missing" check
    catch a foreign document by accident, and the identity check — the one that would still
    fire if every ``WHERE`` clause in storage were deleted — would never be exercised.

    This is what a store written without the ``WHERE`` clause looks like, and it is the only
    way to see the surface's own check fire. A test that used a correct store would pass
    whether or not the check existed.
    """

    @override
    async def list_documents(
        self,
        filter: Filter | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]:
        del filter, limit, offset
        return list(self.documents.values())


@dataclass
class FakeOrganisation:
    """Collections, tags and the trash, held in memory and correctly scoped.

    The control for :class:`LeakyOrganisation`. Every read filters on
    :func:`~manicule.app.tenancy.belongs_to`, exactly as the real store's ``WHERE`` clause
    does, so a refusal seen against the leaky one is a refusal the surface produced rather
    than one every store would have provoked.
    """

    workspace_id: str = "default"
    documents: dict[str, Document] = field(default_factory=dict[str, Document])
    collections: dict[str, DocumentCollection] = field(
        default_factory=dict[str, DocumentCollection]
    )
    members: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    tags: dict[str, Tag] = field(default_factory=dict[str, Tag])
    applied: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    trash: dict[str, Document] = field(default_factory=dict[str, Document])
    restored: list[str] = field(default_factory=list[str])

    async def create_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        rule: CollectionRule | None = None,
    ) -> DocumentCollection:
        label = normalise_name(name)
        if any(item.name == label for item in self.collections.values()):
            msg = f"a collection named {label!r} already exists"
            raise NameInUseError(msg)
        made = DocumentCollection(
            id=f"col-{len(self.collections)}",
            name=label,
            description=description,
            rule=rule,
            created_at=datetime.now(UTC),
        )
        self.collections[made.id] = made
        self.members[made.id] = []
        return made

    async def list_collections(self) -> Sequence[DocumentCollection]:
        return sorted(self.collections.values(), key=lambda item: item.name)

    async def get_collection(self, collection_id: str) -> DocumentCollection | None:
        return self.collections.get(collection_id)

    async def find_collection(self, name: str) -> DocumentCollection | None:
        label = normalise_name(name)
        for item in self.collections.values():
            if item.name == label:
                return item
        return None

    async def rename_collection(self, collection_id: str, name: str) -> DocumentCollection:
        label = normalise_name(name)
        existing = self._require_collection(collection_id)
        rival = await self.find_collection(label)
        if rival is not None and rival.id != collection_id:
            msg = f"a collection named {label!r} already exists"
            raise NameInUseError(msg)
        renamed = existing.model_copy(update={"name": label})
        self.collections[collection_id] = renamed
        return renamed

    async def describe_collection(
        self, collection_id: str, description: str | None
    ) -> DocumentCollection:
        existing = self._require_collection(collection_id)
        described = existing.model_copy(update={"description": description})
        self.collections[collection_id] = described
        return described

    async def collections_for(self, document_id: str) -> Sequence[DocumentCollection]:
        """Manual membership only.

        The rule-driven half is deliberately not modelled here. Evaluating a rule against an
        in-memory dict would be a *second* implementation of ``rule_clause``, and two spellings
        of one rule is the drift that function exists as a single expression to prevent — a
        fake that disagreed with SQL would make these tests agree with the wrong thing. The
        rule half is held against the real store by ``assert_collection_store_contract``.
        """
        return sorted(
            (
                self.collections[identifier]
                for identifier, held in self.members.items()
                if document_id in held and identifier in self.collections
            ),
            key=lambda item: item.name,
        )

    def _require_collection(self, collection_id: str) -> DocumentCollection:
        existing = self.collections.get(collection_id)
        if existing is None:
            msg = f"no collection {collection_id!r}"
            raise UnknownEntityError(msg)
        return existing

    async def delete_collection(self, collection_id: str) -> None:
        if collection_id not in self.collections:
            msg = f"no collection {collection_id!r}"
            raise UnknownEntityError(msg)
        del self.collections[collection_id]
        self.members.pop(collection_id, None)

    async def add_to_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        held = self.members.setdefault(collection_id, [])
        added = 0
        for identifier in document_ids:
            if identifier not in held:
                held.append(identifier)
                added += 1
        return added

    async def remove_from_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        held = self.members.setdefault(collection_id, [])
        removed = 0
        for identifier in document_ids:
            if identifier in held:
                held.remove(identifier)
                removed += 1
        return removed

    async def collection_documents(
        self, collection_id: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        held = [
            self.documents[identifier]
            for identifier in self.members.get(collection_id, [])
            if identifier in self.documents
            and belongs_to(self.workspace_id, self.documents[identifier])
        ]
        return held[offset : offset + limit]

    async def ensure_tag(self, name: str, *, color: str | None = None) -> Tag:
        for existing in self.tags.values():
            if existing.name == name:
                return existing
        made = Tag(id=f"tag-{len(self.tags)}", name=name, color=color)
        self.tags[made.id] = made
        return made

    async def list_tags(self) -> Sequence[Tag]:
        return sorted(self.tags.values(), key=lambda item: item.name)

    async def delete_tag(self, tag_id: str) -> None:
        if tag_id not in self.tags:
            msg = f"no tag {tag_id!r}"
            raise UnknownEntityError(msg)
        del self.tags[tag_id]

    async def tag_document(self, document_id: str, tag_ids: Sequence[str]) -> int:
        held = self.applied.setdefault(document_id, [])
        added = 0
        for identifier in tag_ids:
            if identifier not in self.tags:
                msg = f"no tag {identifier!r}"
                raise UnknownEntityError(msg)
            if identifier not in held:
                held.append(identifier)
                added += 1
        return added

    async def untag_document(self, document_id: str, tag_ids: Sequence[str]) -> int:
        held = self.applied.setdefault(document_id, [])
        removed = 0
        for identifier in tag_ids:
            if identifier in held:
                held.remove(identifier)
                removed += 1
        return removed

    async def tags_for(self, document_id: str) -> Sequence[Tag]:
        return [self.tags[identifier] for identifier in self.applied.get(document_id, [])]

    async def list_trash(
        self, *, grace_s: float, limit: int = 100, offset: int = 0
    ) -> Sequence[TrashEntry]:
        moment = datetime.now(UTC)
        entries = [
            TrashEntry(
                document=document,
                deleted_at=moment,
                purged=False,
                restorable_until=moment + timedelta(seconds=grace_s),
            )
            for document in self.trash.values()
            if belongs_to(self.workspace_id, document)
        ]
        return entries[offset : offset + limit]

    async def restore_document(self, document_id: str) -> Restoration:
        if document_id not in self.trash:
            return Restoration(document_id=document_id, restored=False, reason="not in the trash")
        self.restored.append(document_id)
        del self.trash[document_id]
        return Restoration(
            document_id=document_id, restored=True, reason="restored inside the grace period"
        )


class LeakyOrganisation(FakeOrganisation):
    """Collections and the trash that ignore their workspace scope. **Deliberately broken.**

    The same shape of fault as :class:`LeakyStore`, and for the same reason: the surface's own
    identity check is the second, independent guard, and a test driven only by a correct store
    would pass whether or not that check existed. It ignores the **limit** as well as the
    scope, so a foreign document cannot be caught by accident by a truncation check.
    """

    @override
    async def collection_documents(
        self, collection_id: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        del limit, offset
        return [
            self.documents[identifier]
            for identifier in self.members.get(collection_id, [])
            if identifier in self.documents
        ]

    @override
    async def list_trash(
        self, *, grace_s: float, limit: int = 100, offset: int = 0
    ) -> Sequence[TrashEntry]:
        del grace_s, limit, offset
        moment = datetime.now(UTC)
        return [
            TrashEntry(document=document, deleted_at=moment, purged=False)
            for document in self.trash.values()
        ]


@dataclass
class FakeConversations:
    """Conversations, turns and share links, held in memory.

    The share methods mirror the real store's contract closely enough to be worth stating:
    ``create_share`` re-checks the ceiling and replaces any previous link, and
    ``shared_conversation`` resolves a **hash**, checks expiry and the snapshot boundary, and
    returns citation labels. A fake that resolved a plaintext token or skipped expiry would
    make every sharing test pass over a hole.
    """

    workspace_id: str = "default"
    records: dict[str, ConversationRecord] = field(default_factory=dict[str, ConversationRecord])
    turns: dict[str, list[tuple[Turn, datetime]]] = field(
        default_factory=dict[str, list[tuple[Turn, datetime]]]
    )
    shares: dict[str, tuple[str, datetime, datetime]] = field(
        default_factory=dict[str, tuple[str, datetime, datetime]]
    )
    feedback: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    deleted: list[str] = field(default_factory=list[str])

    def seed(self, conversation_id: str, *turns: Turn) -> str:
        moment = datetime.now(UTC)
        self.records[conversation_id] = ConversationRecord(
            id=conversation_id,
            title="Seeded",
            created_at=moment,
            updated_at=moment,
            messages=len(turns),
        )
        self.turns[conversation_id] = [(turn, moment) for turn in turns]
        return conversation_id

    async def create_conversation(
        self, *, user_id: str | None = None, title: str | None = None
    ) -> str:
        del user_id
        identifier = f"conv_{len(self.records)}"
        moment = datetime.now(UTC)
        self.records[identifier] = ConversationRecord(
            id=identifier, title=title, created_at=moment, updated_at=moment
        )
        self.turns[identifier] = []
        return identifier

    async def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> Sequence[ConversationRecord]:
        ordered = sorted(self.records.values(), key=lambda item: item.id)
        return ordered[offset : offset + limit]

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        return self.records.get(conversation_id)

    async def rename_conversation(self, conversation_id: str, title: str) -> bool:
        record = self.records.get(conversation_id)
        if record is None:
            return False
        self.records[conversation_id] = record.model_copy(update={"title": title})
        return True

    async def soft_delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id not in self.records:
            return False
        self.deleted.append(conversation_id)
        del self.records[conversation_id]
        self.shares.pop(conversation_id, None)
        return True

    async def history(self, conversation_id: str, *, limit: int = 20) -> Sequence[Turn]:
        return [turn for turn, _ in self.turns.get(conversation_id, [])][-limit:]

    async def record_feedback(
        self,
        message_id: str,
        *,
        feedback: Feedback,
        reason: FeedbackReason | None = None,
        comment: str = "",
    ) -> bool:
        del reason, comment
        if not message_id.startswith("msg"):
            return False
        self.feedback.append((message_id, feedback.value))
        return True

    async def create_share(self, link: ShareLink, *, maximum_ttl_s: int) -> bool:
        if link.expires_at > datetime.now(UTC) + timedelta(seconds=maximum_ttl_s):
            msg = f"the share link for {link.conversation_id!r} outlives the ceiling"
            raise ValueError(msg)
        if link.conversation_id not in self.records:
            return False
        self.shares[link.conversation_id] = (
            link.token_hash,
            link.expires_at,
            min(link.shared_at, datetime.now(UTC)),
        )
        return True

    async def revoke_share(self, conversation_id: str) -> bool:
        if conversation_id not in self.records:
            return False
        self.shares.pop(conversation_id, None)
        return True

    async def shared_conversation(
        self, token_hash: str, *, now: datetime, sharing_enabled: bool
    ) -> Sequence[SharedTurn]:
        if not sharing_enabled or not token_hash:
            return []
        for conversation_id, (stored, expires, boundary) in self.shares.items():
            if stored != token_hash or expires <= now:
                continue
            return [
                SharedTurn(
                    role=turn.role,
                    content=turn.content,
                    citations=tuple(redact_for_anonymous(citation) for citation in turn.citations),
                )
                for turn, written in self.turns.get(conversation_id, [])
                if written <= boundary
            ]
        return []


@dataclass
class FakeTelemetry:
    """Query logs and audit entries, recorded in memory.

    ``fails`` makes every write raise, which is what a busy SQLite writer looks like from
    here. It exists so that "a search does not fail because a telemetry insert did" is a
    property a test can watch rather than a claim in a docstring.
    """

    queries: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    audits: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    fails: bool = False

    def _refuse(self) -> None:
        if self.fails:
            msg = "database is locked"
            raise OSError(msg)

    async def record_query(
        self,
        query: str,
        *,
        profile: str,
        chunk_ids: Sequence[str],
        confidence: float | None,
        elapsed_ms: int,
    ) -> str:
        self._refuse()
        identifier = f"q-{len(self.queries)}"
        self.queries.append(
            {
                "id": identifier,
                "query": query,
                "profile": profile,
                "chunks": len(list(chunk_ids)),
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        return identifier

    async def query_logs(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        newest = list(reversed(self.queries))
        return newest[offset : offset + limit], len(self.queries)

    async def record_audit(
        self,
        event_type: str,
        *,
        details: Mapping[str, object],
        actor: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        self._refuse()
        self.audits.append(
            {
                "id": f"a-{len(self.audits)}",
                "event_type": event_type,
                "actor": actor,
                "ip_address": ip_address,
                "details": dict(details),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

    async def audit_logs(
        self, *, limit: int = 50, offset: int = 0, event_type: str | None = None
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        chosen = [
            entry
            for entry in reversed(self.audits)
            if event_type is None or entry["event_type"] == event_type
        ]
        return chosen[offset : offset + limit], len(chosen)


@dataclass
class FakeRetriever:
    """A retriever that returns exactly the candidates it was given."""

    candidates: list[Candidate] = field(default_factory=list[Candidate])
    seen: list[Query] = field(default_factory=list[Query])

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.seen.append(query)
        return RetrievalResult(
            context=Context(query=query, passages=tuple(self.candidates)),
            candidates=list(self.candidates),
            confidence=Confidence(
                score=0.5, band=ConfidenceBand.MEDIUM, reason="a fake, so this is a constant"
            ),
        )


@dataclass
class FakeAnswerer:
    """An answerer that emits one delta and a final envelope."""

    text: str = "The client retries twice."
    calls: list[object] = field(default_factory=list[object])
    envelope: AnswerEnvelope | None = None
    """The final envelope, for a test the default cannot express.

    The default carries **no citations**, which is right for the tests about confidence, streaming
    and tenancy and wrong for anything asserting what a citation *reports* — and the difference was
    invisible until a guard was disabled and nothing went red, because no test on any surface had
    ever constructed an ``AnswerCitation`` through this path. A test needing a real citation sets
    one here rather than reaching past the service, so the payload is assembled by the code that
    assembles it in production.
    """

    def answer(
        self, request: AnswerRequest, result: AnswerResult | None = None
    ) -> AsyncIterator[AnswerEvent]:
        """Record the request, then hand back the stream.

        A plain function returning a generator rather than an async generator function,
        because the two have different *declared* return types and the protocol asks for an
        ``AsyncIterator``. The recording therefore happens when ``answer`` is called rather
        than when the stream is first iterated — which is what
        ``tests/app/test_tenancy.py`` relies on to prove the model was never reached.
        """
        del result
        self.calls.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[AnswerEvent]:
        final = self.envelope or AnswerEnvelope(
            text=self.text, corpus_consulted=True, confidence=0.5
        )
        yield AnswerEvent(kind=EventKind.DELTA, text=final.text)
        yield AnswerEvent(kind=EventKind.FINAL, envelope=final)


@dataclass
class FakeIngestion:
    """An ingest surface that records what it was asked to do."""

    report: RunReport = field(default_factory=lambda: RunReport(connector="local", discovered=1))
    paths: list[Path] = field(default_factory=list[Path])
    synced: list[str] = field(default_factory=list[str])
    reindexed: list[str] = field(default_factory=list[str])
    imported: list[Path] = field(default_factory=list[Path])

    async def index_path(
        self, path: Path, *, name: str, limit: int | None = None, force: bool = False
    ) -> RunReport:
        del name, limit, force
        self.paths.append(path)
        return self.report

    async def sync(self, connector: str, *, limit: int | None = None) -> RunReport:
        del limit
        self.synced.append(connector)
        return self.report

    async def reindex(self, document_id: str) -> ReindexReport:
        self.reindexed.append(document_id)
        return ReindexReport(documents=1, chunks=3)

    async def import_archive(self, path: Path, *, force: bool = False) -> RunReport:
        del force
        self.imported.append(path)
        return self.report


@dataclass
class FakeMaintenance:
    """Whole-installation operations, recorded rather than performed."""

    revision: str | None = "abc123"
    workspace_rows: list[tuple[str, str, str]] = field(
        default_factory=lambda: [("default", "default", "personal")]
    )
    reset: tuple[int, int, bool] = (0, 0, False)
    resets: int = 0
    """How many times the index was actually emptied.

    Counted so a test can assert that a command which *refused* wrote nothing. An exit status
    says the command stopped; it does not say it stopped before doing the thing.
    """
    backups: list[tuple[Path, bool]] = field(default_factory=list[tuple[Path, bool]])
    """Every backup asked for, with the target and whether an insecure one was consented to.

    Recorded because ``--allow-insecure-target`` crosses four layers to reach storage, and a
    flag that parses but never arrives is indistinguishable from one that works right up until
    somebody needs it.
    """
    exports: list[tuple[Path, bool]] = field(default_factory=list[tuple[Path, bool]])
    """Every export asked for, on the same terms and for the same reason as :attr:`backups`."""
    backup_error: Exception | None = None
    """Raised instead of writing, for tests about how a refusal reaches the caller."""

    async def schema_revision(self) -> str | None:
        return self.revision

    async def backup(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> Mapping[str, object]:
        self.backups.append((target, allow_insecure_target))
        if self.backup_error is not None:
            raise self.backup_error
        return {
            "created_at": "2026-01-01T00:00:00Z",
            "files": [],
            "counts": {},
            "path": str(target),
        }

    async def restore(self, source: Path, *, force: bool = False) -> Mapping[str, object]:
        del force
        return {"files": [], "path": str(source)}

    async def reset_index(self) -> tuple[int, int, bool]:
        self.resets += 1
        return self.reset

    async def export_corpus(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> tuple[int, int]:
        self.exports.append((target, allow_insecure_target))
        return 0, 0

    async def workspaces(self) -> Sequence[tuple[str, str, str]]:
        return list(self.workspace_rows)


@dataclass
class FakeKeys:
    """API keys, held in memory."""

    issued: list[ApiKeySummary] = field(default_factory=list[ApiKeySummary])
    workspace: str = "default"
    secrets: dict[str, ApiKeySummary] = field(default_factory=dict[str, ApiKeySummary])
    revoked: set[str] = field(default_factory=set[str])

    async def issue(
        self, name: str, *, role: str, expires_days: int | None = None
    ) -> tuple[ApiKeySummary, str]:
        summary = ApiKeySummary(
            id=f"key-{len(self.issued)}",
            name=name,
            prefix="mnk_abc",
            role=role,
            workspace=self.workspace,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=(
                (datetime.now(UTC) + timedelta(days=expires_days)).isoformat()
                if expires_days
                else None
            ),
        )
        self.issued.append(summary)
        secret = f"mnk_secret_{summary.id}"
        self.secrets[secret] = summary
        return summary, secret

    async def list_keys(self) -> Sequence[ApiKeySummary]:
        return list(self.issued)

    async def revoke(self, name_or_id: str) -> ApiKeySummary:
        for summary in self.issued:
            if name_or_id in {summary.id, summary.name}:
                self.revoked.add(summary.id)
                return summary
        msg = f"no API key named {name_or_id!r} in workspace {self.workspace!r}"
        raise UnknownEntityError(msg)

    async def verify(self, secret: str) -> ApiKeySummary | None:
        """Resolve a secret the same way the real store does: by lookup, then by predicate.

        Revoked and expired keys are refused here rather than merely absent from ``issued``,
        because "the key exists and is no longer usable" is the case a surface test has to be
        able to construct.
        """
        summary = self.secrets.get(secret)
        if summary is None or summary.id in self.revoked:
            return None
        if summary.expires_at and datetime.fromisoformat(summary.expires_at) <= datetime.now(UTC):
            return None
        return summary


@dataclass
class FakeBackend:
    """Everything the service is given, assembled from the fakes above."""

    settings: Settings = field(default_factory=Settings)
    store: FakeStore = field(default_factory=FakeStore)
    retriever_: FakeRetriever = field(default_factory=FakeRetriever)
    answerer_: FakeAnswerer = field(default_factory=FakeAnswerer)
    ingestion_: FakeIngestion = field(default_factory=FakeIngestion)
    maintenance_: FakeMaintenance = field(default_factory=FakeMaintenance)
    organisation_: FakeOrganisation = field(default_factory=FakeOrganisation)
    conversations_: FakeConversations = field(default_factory=FakeConversations)
    telemetry_: FakeTelemetry = field(default_factory=FakeTelemetry)
    keys_: FakeKeys = field(default_factory=FakeKeys)
    discovery: Discovery | None = None
    checks: list[Check] = field(default_factory=list[Check])

    @property
    def workspace(self) -> str:
        return self.settings.workspace

    async def documents(self) -> DocumentSurface:
        return self.store

    async def retriever(self) -> Retrieving:
        return self.retriever_

    async def answerer(self) -> Answering:
        return self.answerer_

    async def ingestion(self) -> Ingesting:
        return self.ingestion_

    async def maintenance(self) -> Maintenance:
        return self.maintenance_

    async def organisation(self) -> Organising:
        return self.organisation_

    async def conversations(self) -> Conversing:
        return self.conversations_

    async def telemetry(self) -> Telemetry:
        return self.telemetry_

    async def keys(self) -> Keys:
        return self.keys_

    async def component_checks(self) -> Sequence[Check]:
        return list(self.checks)


__all__ = [
    "FakeAnswerer",
    "FakeBackend",
    "FakeIngestion",
    "FakeKeys",
    "FakeMaintenance",
    "FakeRetriever",
    "FakeStore",
    "LeakyStore",
    "make_chunk",
    "make_document",
]
