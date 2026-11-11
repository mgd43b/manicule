"""The workspace boundary, checked again at the surface.

The relational store already scopes every query to the workspace its handle carries
(:mod:`manicule.storage.scoped`), and that is where isolation is *enforced*. This module is a
second, independent check on the way **out** of the service, and the reason it exists is that
the two checks cannot fail the same way:

* The store's check is a predicate in a ``WHERE`` clause. It fails open when a query is built
  without it — a new statement, a join added later, a store that a plugin supplied.
* This check is arithmetic on the row that is about to be returned. ``document_id`` derives an
  id from ``(workspace_id, source, source_id)``, so recomputing it from the document's own
  fields and comparing proves the id was **minted for this workspace**. There is no way to
  satisfy it with a document belonging to somebody else, because the digest would not match.

A component that both retrieves and checks is one mistake away from doing neither. The point
of the second check is that it is not the first one written twice: it consults nothing, reads
only the object in hand, and would still fire if every ``WHERE`` clause in the store were
deleted.

The refusal is loud and carries no content from the offending row. A leak reported by quoting
what leaked is still a leak.

**Readers are checked here the same way.** A workspace can restrict documents to some of its
readers (:mod:`manicule.core.readers`), and the store applies that restriction as a predicate
too. :func:`require_readable` is its second check: it reads the gates the document came back
carrying and compares them with the caller's reader set, and consults nothing else. A store that
forgot the predicate, or a path that produced a document without reading its gates, fails it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from manicule.core.errors import ManiculeError
from manicule.core.ids import document_id
from manicule.core.readers import admits

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from manicule.core.readers import DocumentAccess


class CrossWorkspaceError(ManiculeError):
    """Something addressed to one workspace produced another workspace's data.

    Raised rather than filtered. A surface that quietly dropped the foreign rows would answer
    a question nobody asked — "the eleven of these twelve you may see" — and report success.
    """


class Owned(Protocol):
    """The three fields a workspace-scoped identity is derived from."""

    @property
    def id(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def source_id(self) -> str: ...


def belongs_to(workspace: str, document: Owned) -> bool:
    """Whether ``document``'s id was derived for ``workspace``."""
    return document.id == document_id(workspace, document.source, document.source_id)


def require_owned[T: Owned](workspace: str, documents: Iterable[T]) -> Sequence[T]:
    """Return ``documents`` unchanged, having proved every one of them is this tenant's.

    Args:
        workspace: The workspace the request was made in.
        documents: What a store returned.

    Returns:
        The same documents, in the same order.

    Raises:
        CrossWorkspaceError: One of them carries an id that was not derived for this
            workspace. The message names the workspace and how many documents were refused,
            and **never** the offending title, URI or text.
    """
    checked = list(documents)
    foreign = sum(1 for document in checked if not belongs_to(workspace, document))
    if foreign:
        msg = (
            f"{foreign} of {len(checked)} document(s) returned to workspace {workspace!r} "
            f"carry an id that was not derived for it. A document's id is a digest of "
            f"(workspace, source, source_id), so this is a store that ignored its scope or a "
            f"handle opened for the wrong tenant. Nothing was returned."
        )
        raise CrossWorkspaceError(msg)
    return checked


class UnreadableDocumentError(ManiculeError):
    """Something on its way to a caller came from a document that caller may not read.

    Raised rather than filtered, for :class:`CrossWorkspaceError`'s reason: by the time this
    runs, the store has already been asked to exclude the document, so finding one here is a
    defect, and an answer quietly assembled from what was left would be a ranking computed
    over a corpus the caller was never meant to see.
    """


class Gated(Protocol):
    """The one field a reader check reads."""

    @property
    def access(self) -> DocumentAccess | None: ...


def require_readable[T: Gated](
    readers: frozenset[str] | None, documents: Iterable[T]
) -> Sequence[T]:
    """Return ``documents`` unchanged, having proved the caller may read every one of them.

    Args:
        readers: The caller's reader set, ``None`` for a caller no gate applies to.
        documents: What a store returned, each carrying the gates it was read with.

    Returns:
        The same documents, in the same order.

    Raises:
        UnreadableDocumentError: One of them carries a gate the caller's reader set does not
            open, or carries no gates at all because the path that produced it did not read
            them. The message says how many, and **never** which, or what they hold.
    """
    checked = list(documents)
    if readers is None:
        return checked
    refused = sum(1 for document in checked if not admits(readers, document.access))
    if refused:
        msg = (
            f"{refused} of {len(checked)} document(s) on their way to this caller are ones "
            f"the caller may not read. The store was asked to exclude them, so this is a "
            f"store that ignored the reader set or a path that did not read a document's "
            f"gates. Nothing was returned."
        )
        raise UnreadableDocumentError(msg)
    return checked


def require_owns(workspace: str, document: Owned) -> None:
    """Prove one document is this tenant's.

    Raises:
        CrossWorkspaceError: It is not.
    """
    require_owned(workspace, [document])


__all__ = [
    "CrossWorkspaceError",
    "Gated",
    "Owned",
    "UnreadableDocumentError",
    "belongs_to",
    "require_owned",
    "require_owns",
    "require_readable",
]
