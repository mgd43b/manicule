"""Retiring a container's derived documents, which a soft delete does not do by itself.

``documents.container_id`` carries ``ON DELETE CASCADE``, and that cascade runs on a **hard**
delete. Every removal that matters here is a soft one — reconciliation sets ``deleted_at`` and
leaves chunks, vectors and lexical rows in place so a restore inside the grace period is free —
and a timestamp on a parent says nothing about its children.

So a container that goes away leaves its members live, and they are worse than merely stale:
connector-level reconciliation excludes derived documents by design, because their source ids
were never in any inventory and never could be. A member orphaned this way is therefore
searchable, citable, and unreachable by the one pass that would otherwise have removed it.
Retiring the subtree is the only thing that ends it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Document


@runtime_checkable
class ContainerStore(Protocol):
    """The two operations retiring a subtree needs, and nothing else.

    Narrow on purpose: this is reached from the ingest pipeline and from reconciliation, and a
    wider protocol would make a caller that only removes documents look like one that can
    publish them.
    """

    async def container_members(self, container_id: str) -> Sequence[Document]: ...

    async def soft_delete_document(self, document_id: str) -> None: ...


async def retire_subtree(store: ContainerStore, document: Document) -> list[Document]:
    """Soft-delete ``document`` and every live document derived from it, deepest included.

    Iterative rather than recursive, and visited-guarded. A container tree is bounded at three
    levels by ``docs/parsing.md`` §9.2 so the depth is never the problem; the guard is there
    because ``container_id`` is a column and a column can hold whatever a hand-edited database
    or a future migration puts in it, and a cycle would otherwise be a hang rather than a
    finding.

    Returns what was retired, parents before children, so a caller can report it.
    """
    retired: list[Document] = []
    pending = [document]
    seen: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current.id in seen:
            continue
        seen.add(current.id)
        await store.soft_delete_document(current.id)
        retired.append(current)
        pending.extend(await store.container_members(current.id))
    return retired


async def retire_derived(store: ContainerStore, document: Document) -> list[Document]:
    """Soft-delete everything derived from ``document``, leaving ``document`` itself alone.

    For the caller that has already removed the parent on its own terms — connector
    reconciliation removes what a connector stopped reporting, and the members underneath were
    never the connector's to report.
    """
    retired: list[Document] = []
    for member in await store.container_members(document.id):
        retired.extend(await retire_subtree(store, member))
    return retired


__all__ = ["ContainerStore", "retire_derived", "retire_subtree"]
