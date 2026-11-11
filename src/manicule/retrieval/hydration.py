"""The join that turns a set of chunk ids into candidates a query is allowed to see.

One statement, in one place, used by the two paths that can produce chunk ids without having
consulted the authoritative store: the dense leg, whose vector table has no column for tenancy
or liveness, and a cache hit, whose entry was computed against an index that may since have
moved. Both need the same answer to the same question, and two implementations of it would be
two chances to get a security boundary subtly different.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.content import DocumentStatus
from manicule.core.readers import admits

if TYPE_CHECKING:
    from collections.abc import Collection

    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Filter


async def visible_documents(
    docstore: DocStore, join: Filter, document_ids: Collection[str]
) -> dict[str, str]:
    """Which of ``document_ids`` this store will show, under ``join``'s restrictions.

    The store applies the workspace scope and excludes soft-deleted documents; status is
    checked here, because a document mid-ingest is visible to the store and must not be
    visible to a search — its chunks' vectors and text need not agree yet.

    ``join`` carries the whole document-level restriction, so this call is simultaneously the
    tenancy boundary, the reader boundary (:attr:`~manicule.core.retrieval.Filter.readers`) and
    the post-filter for any field that was too numerous to push down.
    That is deliberate: a post-filter written separately would be a second predicate to keep
    in step with the first.
    """
    wanted = frozenset(document_ids)
    if join.document_ids:
        wanted &= join.document_ids
    if not wanted:
        return {}
    documents = await docstore.list_documents(
        join.model_copy(update={"document_ids": wanted}), limit=len(wanted)
    )
    # The reader set is applied twice here, and the second is not the first written again. The
    # store's statement excludes what the caller may not read; the gates each document came back
    # carrying are then checked on the row itself, so a store that dropped the predicate — or
    # one that returned documents without reading their gates — still admits nothing it should
    # not. Dropped rather than raised: this is a join, and an excluded row is what a join drops.
    return {
        document.id: document.publication_id
        for document in documents
        if document.status is DocumentStatus.INDEXED and admits(join.readers, document.access)
    }


__all__ = ["visible_documents"]
