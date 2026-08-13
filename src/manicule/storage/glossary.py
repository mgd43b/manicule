"""Reading and writing scoped glossary entries.

Two operations, and the interesting one is the read.

**The write** replaces a document's entries wholesale, on the same principle
:meth:`~manicule.storage.docstore.SqliteDocStore.replace_chunks` follows: a document is
re-ingested as a whole, so a merge would leave definitions from a version of the page that no
longer exists, still citable, still answering questions.

**The read is a tenancy boundary**, and it is enforced in one statement rather than by
filtering afterwards. Three restrictions apply and each excludes something different:

* the **workspace**, through the join to ``documents`` and this handle's own id;
* **liveness and status**, so a soft-deleted or mid-ingest document contributes no vocabulary
  — its chunks and its vectors need not agree yet, and a definition read out of one would cite
  a passage a search cannot return;
* the **query's own filter**, resolved through :func:`~manicule.storage.organisation.resolve_filter`
  — the same function collection-scoped search resolves through, deliberately, so there is one
  notion of what a collection contains rather than a second one that drifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from sqlalchemy import delete, or_, select

from manicule.core.content import DocumentStatus
from manicule.core.glossary import DefinitionForm, GlossaryEntry
from manicule.core.ids import glossary_entry_id
from manicule.core.retrieval import Filter
from manicule.storage import models
from manicule.storage.organisation import resolve_filter
from manicule.storage.scoped import WorkspaceScoped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from manicule.core.content import Document

VOCABULARY_FILTER_FIELDS = frozenset(
    {
        "workspace_ids",
        "document_ids",
        "sources",
        "media_types",
        "collection_ids",
        "tag_ids",
        "updated_after",
        "updated_before",
    }
)
"""Which of a filter's fields restrict *which glossary a query may consult*.

Every document-level field, and deliberately not ``kinds`` or ``langs``. Those restrict which
**chunks** a search may return, and this lookup returns no chunks — it returns vocabulary. A
query restricted to ``kinds=table`` is asking for table passages, not asking to be kept
ignorant of what an acronym in its own text means, and applying the restriction here would
hide a definition rather than exclude a result.

They are not thereby ignored. The expanded query carries the whole filter into the pipeline
unchanged, and the retriever checks a promoted definition passage against them before it may
be returned — so a chunk-level restriction still decides what comes back, which is what it is
for.

The two fields are **projected out before the guard runs**, exactly as
:func:`~manicule.retrieval.prefilter.join_filter` projects them out before a query over
``documents``. That is the difference between "honoured" and "ignored", and it is the whole
reason this is written as a narrower filter rather than as a larger set of excuses: nothing
here is handed a restriction it then declines to apply.
"""


def vocabulary_filter(source: Filter) -> Filter:
    """The document-level half of ``source``: what a vocabulary lookup can apply.

    The same move :func:`~manicule.retrieval.prefilter.join_filter` makes, for the same reason
    and with the same consequence — the guard below then sees a filter every field of which
    this query really does honour.
    """
    return Filter(
        workspace_ids=source.workspace_ids,
        document_ids=source.document_ids,
        sources=source.sources,
        media_types=source.media_types,
        collection_ids=source.collection_ids,
        tag_ids=source.tag_ids,
        updated_after=source.updated_after,
        updated_before=source.updated_before,
    )


@runtime_checkable
class ListsDocuments(Protocol):
    """The one method this mixin borrows from the class it is combined with.

    Written down rather than reached for, because the borrowing is deliberate: sources, media
    types and update windows are already expressed as a predicate in exactly one place, and a
    second copy inside the glossary lookup would be a second chance to get a restriction subtly
    different — which is how a filter starts admitting rows it was written to exclude. Stating
    the dependency as a protocol makes it checkable; an attribute lookup that happens to work
    would not be.
    """

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol it borrows from
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...


class GlossaryMixin(WorkspaceScoped):
    """Glossary entries, scoped to this handle's workspace.

    Satisfies :class:`~manicule.retrieval.ports.GlossarySource` structurally. It is a mixin on
    :class:`~manicule.storage.scoped.WorkspaceScoped` for the reason every other surface here
    is: one handle, one workspace, one session factory, however many protocols it satisfies.
    """

    async def replace_glossary_entries(
        self, document_id: str, entries: Sequence[GlossaryEntry], *, fingerprint: str
    ) -> None:
        """Make this document's entries exactly ``entries``, and record what produced them.

        Refuses entries attributed to another document rather than silently rewriting their
        ``document_id``: an entry that arrived under the wrong document is a caller bug, and
        correcting it here would file one document's vocabulary under another's scope.

        **The rows and ``documents.glossary_fp`` move in one transaction**, which is what makes
        the lineage a fact about the rows rather than a claim beside them. Two statements in two
        transactions leave a window, and one direction of that window is the failure this
        feature exists to prevent: entries rewritten, fingerprint not yet advanced is merely a
        document that will be repaired again for nothing, but fingerprint advanced and entries
        not rewritten is a document reporting itself current while serving the old detector's
        definitions.

        **An empty ``entries`` still writes the fingerprint.** "The current detector read this
        document and found no definitions" is a derived result and is recorded as one; without
        it, a corpus of ordinary prose would be indistinguishable from a corpus nobody has run a
        detector over, and every sweep would select every document for ever.

        Args:
            document_id: Which document's vocabulary this is.
            entries: Exactly what it states now, possibly nothing.
            fingerprint: Canonical
                :class:`~manicule.core.fingerprints.GlossaryFingerprint` of the detector that
                produced ``entries``.

        Raises:
            ValueError: An entry names a different document.
        """
        foreign = sorted({entry.document_id for entry in entries} - {document_id})
        if foreign:
            named = ", ".join(repr(item) for item in foreign)
            msg = (
                f"glossary entries for document {document_id!r} name document(s) {named}. "
                f"Nothing was written: rewriting the attribution would file one document's "
                f"vocabulary under another's scope, and scope is what decides who can read it."
            )
            raise ValueError(msg)

        async with self._sessions.begin() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                # Not an error. A document that vanished mid-run has nothing to say, and
                # raising here would turn a race into a failed ingest. There is no row to stamp
                # a fingerprint on either, which is the honest outcome: lineage is a column of
                # `documents`, and this document is not one.
                await self._clear_entries(session, document_id)
                return
            await self._clear_entries(session, document_id)
            row.glossary_fp = fingerprint
            aliases: list[tuple[str, str]] = []
            for entry in entries:
                entry_id = glossary_entry_id(entry.chunk_id, entry.acronym, entry.expansion)
                session.add(
                    models.GlossaryEntry(
                        id=entry_id,
                        document_id=document_id,
                        chunk_id=entry.chunk_id,
                        acronym=entry.acronym,
                        display=entry.display,
                        expansion=entry.expansion,
                        location=entry.location,
                        form=entry.form.value,
                        confidence=entry.confidence,
                    )
                )
                aliases += [(entry_id, alias) for alias in dict.fromkeys(entry.aliases)]
            # Flushed before the aliases, explicitly. The unit of work orders inserts by table
            # dependency only where a mapper *relationship* declares one, and there is none here
            # — deliberately, because an alias is a lookup key rather than an object anybody
            # navigates to. Without this the alias row reaches SQLite first and the foreign key
            # rejects it, which is how the parenthetical form's second short form was found to
            # be unwritable at all.
            await session.flush()
            for entry_id, alias in aliases:
                session.add(models.GlossaryAlias(entry_id=entry_id, key=alias))

    async def glossary_lineage(self, document_id: str) -> str | None:
        """Which detector last decided this document's entries, or ``None`` if none has.

        **One column, no join, and that is requirement 3 in one method.** Asking "is this
        document's vocabulary current" must not cost the vocabulary: a corpus-wide answer is a
        scan of an indexed text column, and a scan that dragged in ``glossary_entries`` would
        read every definition in the index to answer a question about none of them.

        The two absent answers are different and neither is inferred from the other. ``None``
        for a document nothing has ever detected against, and a fingerprint whose ``detector``
        reads ``disabled`` for one last ingested while detection was switched off — which is a
        document that may hold perfectly good entries from before the switch.
        """
        async with self._sessions() as session:
            row = await self._live_document(session, document_id)
            return row.glossary_fp if row is not None else None

    async def _clear_entries(self, session: AsyncSession, document_id: str) -> None:
        await session.execute(
            delete(models.GlossaryEntry).where(models.GlossaryEntry.document_id == document_id)
        )

    async def glossary_entries(self, document_id: str) -> Sequence[GlossaryEntry]:
        """Every entry one document states, for diagnostics and for tests.

        Scoped like everything else: a document id from another workspace returns nothing
        rather than that workspace's vocabulary.
        """
        async with self._sessions() as session:
            row = await self._live_document(session, document_id)
            if row is None:
                return []
            rows = (
                (
                    await session.execute(
                        select(models.GlossaryEntry)
                        .where(models.GlossaryEntry.document_id == document_id)
                        .order_by(models.GlossaryEntry.acronym, models.GlossaryEntry.id)
                    )
                )
                .scalars()
                .all()
            )
            return await self._with_aliases(session, rows)

    async def entries_for(
        self,
        keys: Sequence[str],
        filter: Filter,  # noqa: A002 - mirrors the vocabulary every other scoped read uses
    ) -> Sequence[GlossaryEntry]:
        """Every entry in scope whose acronym or alias is one of ``keys``.

        Raises:
            CrossWorkspaceCollisionError: The filter names another workspace.
            ValueError: The filter restricts on a field this lookup cannot honour.
        """
        scoped = vocabulary_filter(filter)
        self._require_honourable(scoped, VOCABULARY_FILTER_FIELDS, "look glossary terms up")
        wanted = [key for key in dict.fromkeys(keys) if key]
        if not wanted:
            return []

        resolved = await resolve_filter(scoped, collections=self, tags=self)  # pyright: ignore[reportArgumentType]
        if resolved is None:
            # The filter named a collection or tag holding nothing. No document is in scope, so
            # no vocabulary is either — collapsing this to "no restriction" would let an empty
            # collection consult the whole workspace's glossary.
            return []

        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.GlossaryEntry)
                        .join(
                            models.Document,
                            models.Document.id == models.GlossaryEntry.document_id,
                        )
                        .outerjoin(
                            models.GlossaryAlias,
                            models.GlossaryAlias.entry_id == models.GlossaryEntry.id,
                        )
                        .where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.deleted_at.is_(None),
                            models.Document.status == DocumentStatus.INDEXED,
                            or_(
                                models.GlossaryEntry.acronym.in_(wanted),
                                models.GlossaryAlias.key.in_(wanted),
                            ),
                        )
                        .order_by(
                            models.GlossaryEntry.acronym,
                            models.GlossaryEntry.confidence.desc(),
                            models.GlossaryEntry.id,
                        )
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                return []
            admitted = await self._admitted_documents(resolved, {row.document_id for row in rows})
            kept = [row for row in rows if row.document_id in admitted]
            return await self._with_aliases(session, kept)

    async def _admitted_documents(
        self,
        resolved: Filter,
        document_ids: set[str],
    ) -> set[str]:
        """Which of ``document_ids`` the query's own filter still admits.

        Asked of :meth:`list_documents` rather than reimplemented as a second ``WHERE`` clause.
        The predicate for sources, media types and update windows already exists in exactly one
        place, and a copy of it here would be a second chance to get a restriction subtly
        different — which is how a filter starts admitting rows it was written to exclude.

        ``kinds`` and ``langs`` are dropped, with the reasoning on
        :data:`VOCABULARY_FILTER_FIELDS`: they restrict chunks, and this returns vocabulary.
        """
        wanted = document_ids
        if resolved.document_ids:
            wanted &= set(resolved.document_ids)
        if not wanted:
            return set()
        scoped = Filter(
            workspace_ids=resolved.workspace_ids,
            document_ids=frozenset(wanted),
            sources=resolved.sources,
            media_types=resolved.media_types,
            updated_after=resolved.updated_after,
            updated_before=resolved.updated_before,
        )
        # Cast rather than inherit: the method belongs to the class this mixin is combined
        # with, and declaring it on the mixin would make every real definition an override
        # of a stub that does nothing.
        lister = cast("ListsDocuments", self)
        found = await lister.list_documents(scoped, limit=len(wanted))
        return {document.id for document in found}

    async def _with_aliases(
        self, session: AsyncSession, rows: Sequence[models.GlossaryEntry]
    ) -> list[GlossaryEntry]:
        if not rows:
            return []
        aliases: dict[str, list[str]] = {}
        found = (
            await session.execute(
                select(models.GlossaryAlias).where(
                    models.GlossaryAlias.entry_id.in_([row.id for row in rows])
                )
            )
        ).scalars()
        for alias in found:
            aliases.setdefault(alias.entry_id, []).append(alias.key)
        return [
            GlossaryEntry(
                acronym=row.acronym,
                display=row.display,
                expansion=row.expansion,
                document_id=row.document_id,
                chunk_id=row.chunk_id,
                location=row.location,
                form=DefinitionForm(row.form),
                confidence=row.confidence,
                aliases=tuple(sorted(aliases.get(row.id, ()))),
            )
            for row in rows
        ]


__all__ = [
    "VOCABULARY_FILTER_FIELDS",
    "GlossaryMixin",
    "ListsDocuments",
    "vocabulary_filter",
]
