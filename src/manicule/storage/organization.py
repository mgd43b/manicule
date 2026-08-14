"""Collections and tags: the two ways a person groups a corpus.

Both are workspace-scoped, both are join tables with cascades, and both refuse to touch a
document this handle cannot see. What separates them is what a duplicate name means. Creating
a collection that already exists is refused, because a collection is a deliberate object and
handing back somebody else's under the same name merges two people's sets. Applying a tag that
already exists is the normal case, so :meth:`TagsMixin.ensure_tag` is idempotent and there is
no strict variant to reach for by mistake.

**Rule-driven membership is evaluated, never materialized.** A collection carrying a
:class:`~manicule.core.organization.CollectionRule` reports what the rule selects *now*. The
clause that expresses the rule is built exactly once, in :func:`rule_clause`, and every reader
uses it — listing a collection's documents, asking which collections a document is in, and
resolving a filter — because two spellings of one rule is a corpus that answers the same
question two ways.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import ColumnElement, Select, and_, delete, false, func, or_, select
from sqlalchemy.exc import IntegrityError

from manicule.core.errors import NameInUseError, UnknownEntityError
from manicule.core.organization import Collection, CollectionRule, Tag
from manicule.storage import models
from manicule.storage.rows import to_document
from manicule.storage.scoped import WorkspaceScoped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from manicule.core.content import Document
    from manicule.core.protocols import CollectionStore, TagStore
    from manicule.core.retrieval import Filter

_WHITESPACE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """The stored form of a collection or tag name.

    Unicode-normalized to NFKC, with surrounding whitespace stripped and internal runs
    collapsed to a single space. Without NFKC the same visible label typed on two keyboards is
    two different rows — a precomposed ``é`` and an ``e`` plus a combining acute are distinct
    byte strings and identical to a reader — and the corpus splits between them with nothing to
    see.

    **Case is preserved, and uniqueness is therefore case-sensitive.** That is a decision, not
    an omission, and it rests on the failure being *visible*: ``Runbook`` and ``runbook`` both
    appear in :meth:`TagsMixin.list_tags`, adjacent, where a person notices them. Compare the
    failure the schema's ``CHECK`` constraints exist for — a misspelled ``documents.status``
    makes a document unservable forever with nothing rendered anywhere. Case-folding is also
    not the free win it looks: ``str.casefold`` maps ``İ`` to two codepoints and folds ``ẞ``
    to ``ss``, so a label would come back spelled differently from how it was typed, and a
    label is display text.

    Raises:
        ValueError: The name is empty once normalized. A nameless label cannot be found again.
    """
    collapsed = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", name)).strip()
    if not collapsed:
        msg = f"name {name!r} is empty once whitespace is collapsed; a label needs a name"
        raise ValueError(msg)
    return collapsed


def rule_clause(rule: CollectionRule) -> ColumnElement[bool]:
    """The SQL predicate a rule stands for, over ``documents``.

    **The single expression of a rule.** Three readers need it — listing a collection,
    reporting a document's collections, and resolving a filter — and a second implementation
    in Python for the "does this one document match" case is how the same rule starts giving
    two answers. Scope is deliberately *not* part of it: the workspace and the soft-delete
    predicate belong to the query, not to the stored rule, which is what stops a saved rule
    ever widening past the handle running it.
    """
    clauses: list[ColumnElement[bool]] = []
    if rule.sources:
        clauses.append(models.Document.source.in_(sorted(rule.sources)))
    if rule.media_types:
        clauses.append(models.Document.media_type.in_(sorted(rule.media_types)))
    if rule.updated_after is not None:
        clauses.append(models.Document.updated_at > rule.updated_after)
    if rule.updated_before is not None:
        clauses.append(models.Document.updated_at < rule.updated_before)
    if rule.tag_ids:
        clauses.append(
            models.Document.id.in_(
                select(models.DocumentTag.document_id).where(
                    models.DocumentTag.tag_id.in_(sorted(rule.tag_ids))
                )
            )
        )
    # `CollectionRule` refuses to be empty, so this always has something in it. `and_` with no
    # arguments renders SQL true, which would silently select the whole workspace — the exact
    # outcome that validator exists to make unreachable.
    return and_(*clauses)


class CollectionsMixin(WorkspaceScoped):
    """:class:`~manicule.core.protocols.CollectionStore` over SQLite."""

    async def create_collection(
        self, name: str, *, description: str | None = None, rule: CollectionRule | None = None
    ) -> Collection:
        """Create a collection, or refuse because the name is taken.

        The check and the insert are in one transaction, and the unique constraint is still
        the authority: two requests naming the same new collection can both find nothing and
        both insert, and the loser must come back with the same sentence as an ordinary
        duplicate rather than an ``IntegrityError`` naming a constraint. A caller cannot act on
        the name of an index.
        """
        label = normalize_name(name)
        try:
            async with self._sessions.begin() as session:
                if await self._collection_named(session, label) is not None:
                    raise NameInUseError(_name_taken(self._workspace_id, "collection", label))
                row = models.Collection(
                    id=str(uuid.uuid4()),
                    workspace_id=self._workspace_id,
                    name=label,
                    description=description,
                    auto_rules=cast("Any", rule.model_dump(mode="json") if rule else None),
                )
                session.add(row)
                await session.flush()
                return _to_collection(row)
        except IntegrityError as clash:
            raise NameInUseError(_name_taken(self._workspace_id, "collection", label)) from clash

    async def get_collection(self, collection_id: str) -> Collection | None:
        async with self._sessions() as session:
            row = await self._collection(session, collection_id)
            return None if row is None else _to_collection(row)

    async def find_collection(self, name: str) -> Collection | None:
        async with self._sessions() as session:
            row = await self._collection_named(session, normalize_name(name))
            return None if row is None else _to_collection(row)

    async def list_collections(self) -> Sequence[Collection]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Collection)
                        .where(models.Collection.workspace_id == self._workspace_id)
                        .order_by(models.Collection.name)
                    )
                )
                .scalars()
                .all()
            )
            return [_to_collection(row) for row in rows]

    async def rename_collection(self, collection_id: str, name: str) -> Collection:
        label = normalize_name(name)
        async with self._sessions.begin() as session:
            row = await self._require_collection(session, collection_id)
            rival = await self._collection_named(session, label)
            if rival is not None and rival.id != row.id:
                msg = (
                    f"{_name_taken(self._workspace_id, 'collection', label)} Renaming onto it "
                    f"would merge two sets under one name."
                )
                raise NameInUseError(msg)
            row.name = label
            await session.flush()
            return _to_collection(row)

    async def describe_collection(self, collection_id: str, description: str | None) -> Collection:
        async with self._sessions.begin() as session:
            row = await self._require_collection(session, collection_id)
            row.description = description
            await session.flush()
            return _to_collection(row)

    async def set_collection_rule(
        self, collection_id: str, rule: CollectionRule | None
    ) -> Collection:
        async with self._sessions.begin() as session:
            row = await self._require_collection(session, collection_id)
            row.auto_rules = cast("Any", rule.model_dump(mode="json") if rule else None)
            await session.flush()
            return _to_collection(row)

    async def delete_collection(self, collection_id: str) -> None:
        """Remove the collection. The cascade takes its memberships and nothing else."""
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.Collection).where(
                    models.Collection.id == collection_id,
                    models.Collection.workspace_id == self._workspace_id,
                )
            )

    async def add_to_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        async with self._sessions.begin() as session:
            await self._require_collection(session, collection_id)
            wanted = await self._require_live_documents(session, document_ids)
            if not wanted:
                return 0
            already = set(
                (
                    await session.execute(
                        select(models.CollectionDocument.document_id).where(
                            models.CollectionDocument.collection_id == collection_id,
                            models.CollectionDocument.document_id.in_(wanted),
                        )
                    )
                )
                .scalars()
                .all()
            )
            added = [document_id for document_id in wanted if document_id not in already]
            for document_id in added:
                session.add(
                    models.CollectionDocument(collection_id=collection_id, document_id=document_id)
                )
            return len(added)

    async def remove_from_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        """Drop manual memberships. A document the rule still selects stays a member."""
        wanted = list(dict.fromkeys(document_ids))
        if not wanted:
            return 0
        async with self._sessions.begin() as session:
            await self._require_collection(session, collection_id)
            present = (
                (
                    await session.execute(
                        select(models.CollectionDocument.document_id).where(
                            models.CollectionDocument.collection_id == collection_id,
                            models.CollectionDocument.document_id.in_(wanted),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not present:
                return 0
            await session.execute(
                delete(models.CollectionDocument).where(
                    models.CollectionDocument.collection_id == collection_id,
                    models.CollectionDocument.document_id.in_(list(present)),
                )
            )
            return len(present)

    async def collection_documents(
        self, collection_id: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        async with self._sessions() as session:
            row = await self._collection(session, collection_id)
            if row is None:
                return []
            rows = (
                (
                    await session.execute(
                        select(models.Document)
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_(None),
                            self._membership_clause(row),
                        )
                        .order_by(models.Document.created_at.desc(), models.Document.id)
                        .limit(limit)
                        .offset(offset)
                    )
                )
                .scalars()
                .all()
            )
            return [to_document(document) for document in rows]

    async def collections_for(self, document_id: str) -> Sequence[Collection]:
        """Every collection holding this document, by hand or by rule.

        The rule-bearing collections are checked with the same clause that lists them, one
        ``EXISTS`` each. There are tens of collections in a workspace, not thousands, and the
        alternative — a second, Python-side reading of the rule — is the drift
        :func:`rule_clause` exists to prevent.
        """
        async with self._sessions() as session:
            if await self._live_document(session, document_id) is None:
                return []
            manual = set(
                (
                    await session.execute(
                        select(models.CollectionDocument.collection_id)
                        .join(
                            models.Collection,
                            models.Collection.id == models.CollectionDocument.collection_id,
                        )
                        .where(
                            models.CollectionDocument.document_id == document_id,
                            models.Collection.workspace_id == self._workspace_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            rows = (
                (
                    await session.execute(
                        select(models.Collection)
                        .where(models.Collection.workspace_id == self._workspace_id)
                        .order_by(models.Collection.name)
                    )
                )
                .scalars()
                .all()
            )

            holding: list[Collection] = []
            for row in rows:
                if row.id in manual:
                    holding.append(_to_collection(row))
                    continue
                rule = _rule_of(row)
                if rule is None:
                    continue
                matched = (
                    await session.execute(
                        select(models.Document.id).where(
                            models.Document.id == document_id,
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_(None),
                            rule_clause(rule),
                        )
                    )
                ).scalar_one_or_none()
                if matched is not None:
                    holding.append(_to_collection(row))
            return holding

    # --- internals --------------------------------------------------------------------------

    def _membership_clause(self, row: models.Collection) -> ColumnElement[bool]:
        """Manual members, plus whatever the rule selects right now."""
        manual = models.Document.id.in_(
            select(models.CollectionDocument.document_id).where(
                models.CollectionDocument.collection_id == row.id
            )
        )
        rule = _rule_of(row)
        # `false()` rather than omitting the branch: a collection with no rule is exactly its
        # manual members, and an omitted disjunct would render as an unrestricted `OR TRUE`.
        return or_(manual, rule_clause(rule) if rule is not None else false())

    async def _collection(
        self, session: AsyncSession, collection_id: str
    ) -> models.Collection | None:
        row = await session.get(models.Collection, collection_id)
        if row is None or row.workspace_id != self._workspace_id:
            return None
        return row

    async def _require_collection(
        self, session: AsyncSession, collection_id: str
    ) -> models.Collection:
        row = await self._collection(session, collection_id)
        if row is None:
            msg = (
                f"no collection {collection_id!r} in workspace {self._workspace_id!r}. "
                f"List the collections this handle serves, or open a store for the workspace "
                f"that owns it."
            )
            raise UnknownEntityError(msg)
        return row

    async def _collection_named(self, session: AsyncSession, name: str) -> models.Collection | None:
        return (
            await session.execute(
                select(models.Collection).where(
                    models.Collection.workspace_id == self._workspace_id,
                    models.Collection.name == name,
                )
            )
        ).scalar_one_or_none()


class TagsMixin(WorkspaceScoped):
    """:class:`~manicule.core.protocols.TagStore` over SQLite."""

    async def ensure_tag(self, name: str, *, color: str | None = None) -> Tag:
        """Return the tag with this name, creating it if it is new.

        **The race is resolved rather than reported**, and that is the difference between this
        and :meth:`CollectionsMixin.create_collection`. Two requests applying the same new label
        can both find nothing and both insert; the loser has still got what it asked for — a tag
        with that name — so handing it back is the honest answer, where refusing a *collection*
        is, because a collection is a set somebody is building and a tag is a word.
        """
        label = normalize_name(name)
        try:
            async with self._sessions.begin() as session:
                existing = await self._tag_named(session, label)
                if existing is not None:
                    # The color is deliberately not overwritten. Otherwise the last person to
                    # type a tag name decides how it looks for everyone, from a call that reads
                    # like a no-op.
                    return _to_tag(existing)
                row = models.Tag(
                    id=str(uuid.uuid4()),
                    workspace_id=self._workspace_id,
                    name=label,
                    color=color,
                )
                session.add(row)
                await session.flush()
                return _to_tag(row)
        except IntegrityError:
            settled = await self.find_tag(label)
            if settled is None:  # pragma: no cover - the constraint said it was there
                raise
            return settled

    async def get_tag(self, tag_id: str) -> Tag | None:
        async with self._sessions() as session:
            row = await self._tag(session, tag_id)
            return None if row is None else _to_tag(row)

    async def find_tag(self, name: str) -> Tag | None:
        async with self._sessions() as session:
            row = await self._tag_named(session, normalize_name(name))
            return None if row is None else _to_tag(row)

    async def list_tags(self) -> Sequence[Tag]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Tag)
                        .where(models.Tag.workspace_id == self._workspace_id)
                        .order_by(models.Tag.name)
                    )
                )
                .scalars()
                .all()
            )
            return [_to_tag(row) for row in rows]

    async def rename_tag(self, tag_id: str, name: str) -> Tag:
        label = normalize_name(name)
        async with self._sessions.begin() as session:
            row = await self._require_tag(session, tag_id)
            rival = await self._tag_named(session, label)
            if rival is not None and rival.id != row.id:
                msg = (
                    f"{_name_taken(self._workspace_id, 'tag', label)} Renaming onto it would "
                    f"move every document from one label to the other with nothing to undo it; "
                    f"untag and re-tag if that is what you mean."
                )
                raise NameInUseError(msg)
            row.name = label
            await session.flush()
            return _to_tag(row)

    async def set_tag_color(self, tag_id: str, color: str | None) -> Tag:
        async with self._sessions.begin() as session:
            row = await self._require_tag(session, tag_id)
            row.color = color
            await session.flush()
            return _to_tag(row)

    async def delete_tag(self, tag_id: str) -> None:
        """Remove the tag. The cascade takes its applications and nothing else."""
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.Tag).where(
                    models.Tag.id == tag_id,
                    models.Tag.workspace_id == self._workspace_id,
                )
            )

    async def tag_document(self, document_id: str, tag_ids: Sequence[str]) -> int:
        async with self._sessions.begin() as session:
            await self._require_live_documents(session, [document_id])
            wanted = await self._require_tags(session, tag_ids)
            if not wanted:
                return 0
            already = set(
                (
                    await session.execute(
                        select(models.DocumentTag.tag_id).where(
                            models.DocumentTag.document_id == document_id,
                            models.DocumentTag.tag_id.in_(wanted),
                        )
                    )
                )
                .scalars()
                .all()
            )
            added = [tag_id for tag_id in wanted if tag_id not in already]
            for tag_id in added:
                session.add(models.DocumentTag(document_id=document_id, tag_id=tag_id))
            return len(added)

    async def untag_document(self, document_id: str, tag_ids: Sequence[str]) -> int:
        wanted = list(dict.fromkeys(tag_ids))
        if not wanted:
            return 0
        async with self._sessions.begin() as session:
            await self._require_live_documents(session, [document_id])
            present = (
                (
                    await session.execute(
                        select(models.DocumentTag.tag_id).where(
                            models.DocumentTag.document_id == document_id,
                            models.DocumentTag.tag_id.in_(wanted),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not present:
                return 0
            await session.execute(
                delete(models.DocumentTag).where(
                    models.DocumentTag.document_id == document_id,
                    models.DocumentTag.tag_id.in_(list(present)),
                )
            )
            return len(present)

    async def tags_for(self, document_id: str) -> Sequence[Tag]:
        async with self._sessions() as session:
            if await self._live_document(session, document_id) is None:
                return []
            rows = (
                (
                    await session.execute(
                        select(models.Tag)
                        .join(models.DocumentTag, models.DocumentTag.tag_id == models.Tag.id)
                        .where(
                            models.DocumentTag.document_id == document_id,
                            models.Tag.workspace_id == self._workspace_id,
                        )
                        .order_by(models.Tag.name)
                    )
                )
                .scalars()
                .all()
            )
            return [_to_tag(row) for row in rows]

    async def documents_with_tags(
        self, tag_ids: Sequence[str], *, match_all: bool = False, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        wanted = list(dict.fromkeys(tag_ids))
        if not wanted:
            return []
        async with self._sessions() as session:
            statement = (
                select(models.Document)
                .where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.deleted_at.is_(None),
                    models.Document.id.in_(_tagged_ids(wanted, match_all=match_all)),
                )
                .order_by(models.Document.created_at.desc(), models.Document.id)
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(statement)).scalars().all()
            return [to_document(row) for row in rows]

    # --- internals --------------------------------------------------------------------------

    async def _tag(self, session: AsyncSession, tag_id: str) -> models.Tag | None:
        row = await session.get(models.Tag, tag_id)
        if row is None or row.workspace_id != self._workspace_id:
            return None
        return row

    async def _require_tag(self, session: AsyncSession, tag_id: str) -> models.Tag:
        row = await self._tag(session, tag_id)
        if row is None:
            msg = f"no tag {tag_id!r} in workspace {self._workspace_id!r}."
            raise UnknownEntityError(msg)
        return row

    async def _require_tags(self, session: AsyncSession, tag_ids: Sequence[str]) -> list[str]:
        wanted = list(dict.fromkeys(tag_ids))
        if not wanted:
            return []
        found = set(
            (
                await session.execute(
                    select(models.Tag.id).where(
                        models.Tag.id.in_(wanted),
                        models.Tag.workspace_id == self._workspace_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        missing = [tag_id for tag_id in wanted if tag_id not in found]
        if missing:
            named = ", ".join(repr(tag_id) for tag_id in missing)
            msg = (
                f"no tag {named} in workspace {self._workspace_id!r}. Nothing was written: a "
                f"partially applied tagging reports a success it did not have."
            )
            raise UnknownEntityError(msg)
        return wanted

    async def _tag_named(self, session: AsyncSession, name: str) -> models.Tag | None:
        return (
            await session.execute(
                select(models.Tag).where(
                    models.Tag.workspace_id == self._workspace_id,
                    models.Tag.name == name,
                )
            )
        ).scalar_one_or_none()


async def resolve_filter(
    filter: Filter,  # noqa: A002 - mirrors the vocabulary of the domain
    *,
    collections: CollectionStore,
    tags: TagStore,
) -> Filter | None:
    """Turn ``collection_ids`` and ``tag_ids`` into ``document_ids``, or refuse the query.

    Both stores refuse a filter naming those fields (``LISTABLE_FILTER_FIELDS``), because
    neither the lexical statement nor the vector predicate has a join to reach them. This is
    the resolution step that comes first, and it belongs beside the join tables rather than
    inside a retrieval stage.

    **``None`` means "no document can match", and it is the whole reason this returns an
    option.** A filter's set-valued fields default to empty, and an empty field restricts
    nothing — so resolving an empty collection into ``document_ids=frozenset()`` would produce
    a filter that searches the entire workspace. The restriction would not merely be lost, it
    would inverted: the narrowest possible request answered with the widest possible result,
    ranked and plausible. A caller that ignores ``None`` and passes the original filter through
    gets exactly that, which is why the option is in the signature rather than an empty result
    in the return.

    Fields combine as :class:`~manicule.core.retrieval.Filter` says they do — disjunction
    within a field, conjunction between them. So several collections union, several tags union,
    and a filter naming both keeps only the documents in both sets.

    Raises:
        ValueError: A collection or a tag holds more documents than
            :data:`MAX_RESOLVED_DOCUMENTS`. Refused rather than truncated: a truncated id set
            is a filter that *looks* complete and quietly excludes documents that are in the
            collection, which is the same silent wrongness the ``None`` above exists to
            prevent, arriving from the opposite direction. The remedy is the id-list threshold
            ``docs/retrieval.md`` §3.3 already owns — over-fetch and post-filter instead of a
            million-element ``IN`` clause.
    """
    if not filter.collection_ids and not filter.tag_ids:
        return filter

    scopes: list[set[str]] = []
    if filter.collection_ids:
        from_collections: set[str] = set()
        for collection_id in sorted(filter.collection_ids):
            members = await collections.collection_documents(
                collection_id, limit=MAX_RESOLVED_DOCUMENTS + 1
            )
            _refuse_if_truncated(len(members), f"collection {collection_id!r}")
            from_collections.update(document.id for document in members)
        _refuse_if_truncated(len(from_collections), "the named collections together")
        scopes.append(from_collections)
    if filter.tag_ids:
        tagged = await tags.documents_with_tags(
            sorted(filter.tag_ids), limit=MAX_RESOLVED_DOCUMENTS + 1
        )
        _refuse_if_truncated(len(tagged), "the named tags")
        scopes.append({document.id for document in tagged})
    if filter.document_ids:
        scopes.append(set(filter.document_ids))

    # Intersected by hand rather than through `set.intersection(*scopes)`, which loses the
    # element type. The list is never empty here: the guard above returned already if neither
    # collection nor tag field restricted anything.
    resolved = scopes[0]
    for scope in scopes[1:]:
        resolved &= scope
    if not resolved:
        return None
    return filter.model_copy(
        update={
            "document_ids": frozenset(resolved),
            "collection_ids": frozenset(),
            "tag_ids": frozenset(),
        }
    )


MAX_RESOLVED_DOCUMENTS = 10_000
"""How many document ids a membership resolution will produce before it refuses.

A real ceiling, not a synonym for infinity, and the reason it is enforced rather than merely
documented: past it, the resolution would hand back a filter that looks complete and silently
omits documents that *are* in the collection. That is the same failure as resolving an empty
collection to "no restriction", from the other end.

Refusing is also the only answer SQLite can act on. A resolved set reaches the lexical query as
one bind parameter per id, against a ``SQLITE_MAX_VARIABLE_NUMBER`` that is 32 766 on a modern
build and 999 on an older one — so a large ``IN`` list does not degrade, it fails, and it fails
somewhere that reads as a bug in search. The regime that serves a collection this size is
``docs/retrieval.md`` §3.3's other plan: over-fetch and post-filter, decided per query against
``prefilter_id_limit``, which starts two orders of magnitude below this number.
"""


def _refuse_if_truncated(found: int, what: str) -> None:
    if found <= MAX_RESOLVED_DOCUMENTS:
        return
    msg = (
        f"{what} holds more than {MAX_RESOLVED_DOCUMENTS} documents, which is more than a "
        f"filter can carry as an id list. Truncating it would return a filter that looks "
        f"complete while quietly excluding members of the collection, so this is refused "
        f"instead. To search it now: drop the collection and restrict by source, media type "
        f"or date, or split the collection into smaller ones — membership is metadata, so "
        f"splitting re-indexes nothing and re-embeds nothing."
    )
    raise ValueError(msg)


def _name_taken(workspace_id: str, kind: str, name: str) -> str:
    """One sentence for a name clash, so the pre-check and the constraint agree."""
    return f"workspace {workspace_id!r} already has a {kind} called {name!r}."


def _tagged_ids(tag_ids: Sequence[str], *, match_all: bool) -> Select[tuple[str]]:
    """Document ids carrying any of these tags, or all of them.

    ``match_all`` counts distinct tags per document rather than joining ``document_tags`` once
    per tag. A join per tag turns "documents carrying all five of these" into a five-way
    self-join whose plan degrades with the number of tags; a ``GROUP BY`` with a ``HAVING``
    does not, and it reads as the question being asked.
    """
    applications = select(models.DocumentTag.document_id).where(
        models.DocumentTag.tag_id.in_(list(tag_ids))
    )
    if not match_all:
        return applications
    return applications.group_by(models.DocumentTag.document_id).having(
        func.count(models.DocumentTag.tag_id.distinct()) == len(tag_ids)
    )


def _rule_of(row: models.Collection) -> CollectionRule | None:
    """The stored rule, validated on the way out.

    Validated rather than trusted, on the same terms as ``chunks.anchor``: a JSON column whose
    shape is known is checked at both ends, so a hand-edited row fails where it is read instead
    of quietly selecting a different set of documents.
    """
    stored = row.auto_rules
    if not stored:
        return None
    return CollectionRule.model_validate(stored)


def _to_collection(row: models.Collection) -> Collection:
    return Collection(
        id=row.id,
        name=row.name,
        description=row.description,
        rule=_rule_of(row),
        created_at=row.created_at,
    )


def _to_tag(row: models.Tag) -> Tag:
    return Tag(id=row.id, name=row.name, color=row.color)


__all__ = [
    "MAX_RESOLVED_DOCUMENTS",
    "CollectionsMixin",
    "TagsMixin",
    "normalize_name",
    "resolve_filter",
    "rule_clause",
]
