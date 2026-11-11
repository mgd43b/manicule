"""Reader groups, source grants, and the predicate that applies them to ``documents``.

An administrator makes a reader group, puts people in it, and grants a source to it. From then
on a document of that source is readable only by members of a granted group; a source with no
grant stays readable by every reader of the workspace, which is what every source was before
any of this existed.

**The grant is evaluated when a query runs, never copied onto documents.** A copy would be a
second value that can disagree with the first — the reason the vector store carries no tenancy
column — and it would make a revocation wait for the next sync. So the predicate below is a
correlated subquery on ``documents.source``, and it is written in two spellings that must say
the same thing: :func:`readable` for statements SQLAlchemy builds, and :func:`readable_sql` for
the lexical leg's textual statement. ``tests/test_storage_readers.py`` holds the two to one
answer over the same rows.

**The gates come back with the document.** :func:`access_for` reads them in one statement for a
page of documents, and the store attaches them to each :class:`~manicule.core.content.Document`
it returns from :meth:`~manicule.storage.docstore.SqliteDocStore.list_documents` and
:meth:`~manicule.storage.docstore.SqliteDocStore.get_document`. That is what lets the service
re-check readability on the row in hand, after the store's ``WHERE`` clause, without consulting
anything — the same shape as :func:`~manicule.app.tenancy.require_owned`.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ColumnElement, delete, exists, false, func, or_, select
from sqlalchemy.exc import IntegrityError

from manicule.core.content import DocumentStatus
from manicule.core.errors import NameInUseError, PolicyError, UnknownEntityError
from manicule.core.readers import (
    UNRESTRICTED,
    AccessGate,
    DocumentAccess,
    GateOrigin,
    ReadableTotals,
    ReaderGroup,
    group_ids,
    group_principal,
    user_principal,
)
from manicule.storage import models
from manicule.storage.organization import normalize_name
from manicule.storage.scoped import WorkspaceScoped

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


def readable(readers: frozenset[str]) -> ColumnElement[bool]:
    """Documents a caller holding ``readers`` may read, as a clause over ``documents``.

    A document passes when its source carries no grant, or when one of its source's grants
    names a group among ``readers``. Correlated on ``documents``, so it composes with any
    statement that selects from that table.
    """
    grant = models.SourceReaderGrant
    granted = (
        grant.workspace_id == models.Document.workspace_id,
        grant.source == models.Document.source,
    )
    held = sorted(group_ids(readers))
    admitted: ColumnElement[bool] = (
        exists().where(*granted, grant.group_id.in_(held)) if held else false()
    )
    return or_(~exists().where(*granted), admitted)


_UNGRANTED_SQL = (
    "NOT EXISTS (SELECT 1 FROM source_reader_grants g "
    "WHERE g.workspace_id = d.workspace_id AND g.source = d.source)"
)
_GRANTED_SQL = (
    "EXISTS (SELECT 1 FROM source_reader_grants g "
    "WHERE g.workspace_id = d.workspace_id AND g.source = d.source AND g.group_id IN ({held}))"
)


def readable_sql(readers: frozenset[str]) -> tuple[str, dict[str, Any]]:
    """:func:`readable` as a fragment of textual SQL over ``documents`` aliased ``d``.

    Returns the fragment, to be preceded by ``AND``, and the parameters it binds. Every value
    is a bound parameter; only the parameters' generated names are spliced in.
    """
    held = sorted(group_ids(readers))
    names = [f"reader_group_{index}" for index in range(len(held))]
    if not held:
        return f"({_UNGRANTED_SQL})", {}
    granted = _GRANTED_SQL.format(held=", ".join(":" + name for name in names))
    return f"({_UNGRANTED_SQL} OR {granted})", dict(zip(names, held, strict=True))


async def access_for(
    session: AsyncSession, workspace_id: str, documents: Iterable[models.Document]
) -> dict[str, DocumentAccess]:
    """The gates each of ``documents`` carries, keyed by document id. One statement."""
    rows = list(documents)
    sources = sorted({row.source for row in rows})
    if not sources:
        return {}
    grant = models.SourceReaderGrant
    granted: dict[str, set[str]] = {}
    for source, group_id in (
        await session.execute(
            select(grant.source, grant.group_id).where(
                grant.workspace_id == workspace_id, grant.source.in_(sources)
            )
        )
    ).all():
        granted.setdefault(str(source), set()).add(group_principal(str(group_id)))
    access: dict[str, DocumentAccess] = {}
    for row in rows:
        principals = granted.get(row.source)
        access[row.id] = (
            UNRESTRICTED
            if principals is None
            else DocumentAccess(
                gates=(AccessGate(origin=GateOrigin.SOURCE, readers=frozenset(principals)),)
            )
        )
    return access


class ReadersMixin(WorkspaceScoped):
    """Reader groups and source grants for one workspace.

    Every statement carries this handle's workspace. A person is installation-wide, so a
    member is named by ``users.id``; whether that person belongs to this workspace is the
    caller's to check, because the membership table belongs to the people store.
    """

    async def has_reader_gates(self) -> bool:
        """Whether any document of this workspace could be refused to any of its readers."""
        async with self._sessions() as session:
            found = await session.scalar(
                select(models.SourceReaderGrant.source)
                .where(models.SourceReaderGrant.workspace_id == self._workspace_id)
                .limit(1)
            )
        return found is not None

    async def reader_principals(self, user_id: str) -> frozenset[str]:
        """Everything ``user_id`` holds in this workspace: themself, and each group they are in."""
        async with self._sessions() as session:
            held = (
                (
                    await session.execute(
                        select(models.ReaderGroupMember.group_id)
                        .join(
                            models.ReaderGroup,
                            models.ReaderGroup.id == models.ReaderGroupMember.group_id,
                        )
                        .where(
                            models.ReaderGroupMember.user_id == user_id,
                            models.ReaderGroup.workspace_id == self._workspace_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
        return frozenset({user_principal(user_id), *(group_principal(g) for g in held)})

    async def create_reader_group(self, name: str) -> ReaderGroup:
        """Make an empty reader group.

        Raises:
            NameInUseError: This workspace already has a group by that name.
        """
        label = normalize_name(name)
        row = models.ReaderGroup(id=str(uuid.uuid4()), workspace_id=self._workspace_id, name=label)
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            msg = f"workspace {self._workspace_id!r} already has a reader group named {label!r}"
            raise NameInUseError(msg) from exc
        return ReaderGroup(id=row.id, name=label)

    async def list_reader_groups(self) -> Sequence[ReaderGroup]:
        """Every reader group here, by name, with its members and the sources granted to it."""
        async with self._sessions() as session:
            groups = (
                (
                    await session.execute(
                        select(models.ReaderGroup)
                        .where(models.ReaderGroup.workspace_id == self._workspace_id)
                        .order_by(models.ReaderGroup.name)
                    )
                )
                .scalars()
                .all()
            )
            return [await self._summary(session, row) for row in groups]

    async def find_reader_group(self, name_or_id: str) -> ReaderGroup | None:
        """The group with this id or this name, here, or ``None``."""
        async with self._sessions() as session:
            row = await self._group(session, name_or_id)
            return None if row is None else await self._summary(session, row)

    async def delete_reader_group(self, name_or_id: str) -> ReaderGroup:
        """Delete a group that no source is granted to.

        Raises:
            UnknownEntityError: No such group here.
            PolicyError: A source is still granted to it. Revoke the grant first: deleting the
                last grant on a source opens it to every reader, and that has to be a decision
                made about the source rather than a side effect of tidying up a group.
        """
        async with self._sessions.begin() as session:
            row = await self._require_group(session, name_or_id)
            summary = await self._summary(session, row)
            if summary.sources:
                named = ", ".join(repr(source) for source in summary.sources)
                msg = (
                    f"reader group {summary.name!r} is granted source(s) {named}. Revoke them "
                    f"first; a source left with no grant is readable by everyone here, so that "
                    f"has to be decided about the source, not implied by deleting a group."
                )
                raise PolicyError(msg)
            await session.delete(row)
        return summary

    async def add_reader_group_members(
        self, name_or_id: str, user_ids: Sequence[str]
    ) -> ReaderGroup:
        """Put people in a group. Idempotent for someone already in it.

        Raises:
            UnknownEntityError: No such group here.
        """
        async with self._sessions.begin() as session:
            row = await self._require_group(session, name_or_id)
            present = set(
                (
                    await session.execute(
                        select(models.ReaderGroupMember.user_id).where(
                            models.ReaderGroupMember.group_id == row.id
                        )
                    )
                )
                .scalars()
                .all()
            )
            for user_id in dict.fromkeys(user_ids):
                if user_id not in present:
                    session.add(models.ReaderGroupMember(group_id=row.id, user_id=user_id))
            await session.flush()
            return await self._summary(session, row)

    async def remove_reader_group_members(
        self, name_or_id: str, user_ids: Sequence[str]
    ) -> ReaderGroup:
        """Take people out of a group. Removing someone who is not in it changes nothing.

        Raises:
            UnknownEntityError: No such group here.
        """
        async with self._sessions.begin() as session:
            row = await self._require_group(session, name_or_id)
            await session.execute(
                delete(models.ReaderGroupMember).where(
                    models.ReaderGroupMember.group_id == row.id,
                    models.ReaderGroupMember.user_id.in_(list(user_ids)),
                )
            )
            return await self._summary(session, row)

    async def grant_source(self, source: str, groups: Sequence[str]) -> tuple[str, ...]:
        """Grant ``source`` to each of ``groups``, and return every group it is now granted to.

        Raises:
            UnknownEntityError: A group named is not one of this workspace's. Nothing is
                granted: a grant applied to some of the groups named would narrow the source
                for everyone but the people in the group nobody could find.
        """
        async with self._sessions.begin() as session:
            rows = [await self._require_group(session, group) for group in groups]
            present = set(await self._granted(session, source))
            for row in rows:
                if row.id not in present:
                    session.add(
                        models.SourceReaderGrant(
                            workspace_id=self._workspace_id, source=source, group_id=row.id
                        )
                    )
                    present.add(row.id)
            await session.flush()
            return await self._granted_names(session, source)

    async def revoke_source(self, source: str, groups: Sequence[str]) -> tuple[str, ...]:
        """Revoke ``source`` from each of ``groups``, and return the groups it is still granted to.

        An empty result means the source is now readable by every reader of the workspace.

        Raises:
            UnknownEntityError: A group named is not one of this workspace's.
        """
        async with self._sessions.begin() as session:
            rows = [await self._require_group(session, group) for group in groups]
            await session.execute(
                delete(models.SourceReaderGrant).where(
                    models.SourceReaderGrant.workspace_id == self._workspace_id,
                    models.SourceReaderGrant.source == source,
                    models.SourceReaderGrant.group_id.in_([row.id for row in rows]),
                )
            )
            return await self._granted_names(session, source)

    async def source_grants(self) -> dict[str, tuple[str, ...]]:
        """Every granted source here, and the names of the groups it is granted to."""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(models.SourceReaderGrant.source, models.ReaderGroup.name)
                    .join(
                        models.ReaderGroup,
                        models.ReaderGroup.id == models.SourceReaderGrant.group_id,
                    )
                    .where(models.SourceReaderGrant.workspace_id == self._workspace_id)
                    .order_by(models.SourceReaderGrant.source, models.ReaderGroup.name)
                )
            ).all()
        granted: dict[str, list[str]] = {}
        for source, name in rows:
            granted.setdefault(str(source), []).append(str(name))
        return {source: tuple(names) for source, names in granted.items()}

    async def readable_document_ids(
        self, document_ids: Iterable[str], readers: frozenset[str]
    ) -> frozenset[str]:
        """Which of ``document_ids`` are this workspace's and readable by ``readers``.

        In any state, the trash included: the by-id operations that act on a document — tag
        it, file it, delete it, restore it — must not act on one the caller may not read, and
        must not tell them it exists by refusing differently.
        """
        wanted = sorted(set(document_ids))
        if not wanted:
            return frozenset()
        async with self._sessions() as session:
            found = (
                (
                    await session.execute(
                        select(models.Document.id).where(
                            models.Document.workspace_id == self._workspace_id,
                            models.Document.id.in_(wanted),
                            readable(readers),
                        )
                    )
                )
                .scalars()
                .all()
            )
        return frozenset(str(found_id) for found_id in found)

    async def may_read_source(self, source: str, readers: frozenset[str]) -> bool:
        """Whether ``readers`` may read a document of ``source`` here — the same rule as
        :func:`readable`, asked of a source rather than of its documents."""
        async with self._sessions() as session:
            granted = set(await self._granted(session, source))
        return not granted or bool(granted & group_ids(readers))

    async def readable_totals(self, readers: frozenset[str]) -> ReadableTotals:
        """How much of this workspace a caller holding ``readers`` may read, counted now.

        What every count surface reports to a restricted caller, instead of the workspace's own
        totals: a count of documents somebody may not read says those documents exist, and how
        many of them there are, which is a disclosure of its own.
        """
        live = (
            models.Document.workspace_id == self._workspace_id,
            models.Document.deleted_at.is_(None),
            readable(readers),
        )
        async with self._sessions() as session:
            documents = int(
                (
                    await session.execute(
                        select(func.count()).select_from(models.Document).where(*live)
                    )
                ).scalar_one()
            )
            chunks = int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(models.Chunk)
                        .join(models.Document, models.Document.id == models.Chunk.document_id)
                        .where(*live)
                    )
                ).scalar_one()
            )
            grouped: dict[str, dict[str, int]] = {}
            for label, column in (
                ("by_source", models.Document.source),
                ("by_media_type", models.Document.media_type),
                ("by_status", models.Document.status),
            ):
                rows = (
                    await session.execute(
                        select(column, func.count()).where(*live).group_by(column)
                    )
                ).all()
                grouped[label] = {
                    (value.value if isinstance(value, DocumentStatus) else str(value)): int(count)
                    for value, count in rows
                }
        return ReadableTotals(
            documents=documents,
            chunks=chunks,
            by_source=grouped["by_source"],
            by_media_type=grouped["by_media_type"],
            by_status=grouped["by_status"],
        )

    # --- helpers ----------------------------------------------------------------------------

    async def _group(self, session: AsyncSession, name_or_id: str) -> models.ReaderGroup | None:
        by_id = await session.get(models.ReaderGroup, name_or_id)
        if by_id is not None and by_id.workspace_id == self._workspace_id:
            return by_id
        try:
            label = normalize_name(name_or_id)
        except ValueError:
            return None
        return await session.scalar(
            select(models.ReaderGroup).where(
                models.ReaderGroup.workspace_id == self._workspace_id,
                models.ReaderGroup.name == label,
            )
        )

    async def _require_group(self, session: AsyncSession, name_or_id: str) -> models.ReaderGroup:
        row = await self._group(session, name_or_id)
        if row is None:
            msg = f"no reader group {name_or_id!r} in workspace {self._workspace_id!r}"
            raise UnknownEntityError(msg)
        return row

    async def _granted(self, session: AsyncSession, source: str) -> list[str]:
        return [
            str(group_id)
            for group_id in (
                await session.execute(
                    select(models.SourceReaderGrant.group_id).where(
                        models.SourceReaderGrant.workspace_id == self._workspace_id,
                        models.SourceReaderGrant.source == source,
                    )
                )
            )
            .scalars()
            .all()
        ]

    async def _granted_names(self, session: AsyncSession, source: str) -> tuple[str, ...]:
        names = (
            (
                await session.execute(
                    select(models.ReaderGroup.name)
                    .join(
                        models.SourceReaderGrant,
                        models.SourceReaderGrant.group_id == models.ReaderGroup.id,
                    )
                    .where(
                        models.SourceReaderGrant.workspace_id == self._workspace_id,
                        models.SourceReaderGrant.source == source,
                    )
                    .order_by(models.ReaderGroup.name)
                )
            )
            .scalars()
            .all()
        )
        return tuple(str(name) for name in names)

    async def _summary(self, session: AsyncSession, row: models.ReaderGroup) -> ReaderGroup:
        members = (
            (
                await session.execute(
                    select(models.ReaderGroupMember.user_id)
                    .where(models.ReaderGroupMember.group_id == row.id)
                    .order_by(models.ReaderGroupMember.user_id)
                )
            )
            .scalars()
            .all()
        )
        sources = (
            (
                await session.execute(
                    select(models.SourceReaderGrant.source)
                    .where(models.SourceReaderGrant.group_id == row.id)
                    .order_by(models.SourceReaderGrant.source)
                )
            )
            .scalars()
            .all()
        )
        return ReaderGroup(
            id=row.id,
            name=row.name,
            members=tuple(str(member) for member in members),
            sources=tuple(str(source) for source in sources),
        )


__all__ = ["ReadersMixin", "access_for", "readable", "readable_sql"]
