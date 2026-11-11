"""Who may read a document: the caller's reader set, and the gates a document carries.

The workspace is one boundary and it is coarse on purpose: everyone a workspace admits can
search all of it. Readers are a second boundary *inside* a workspace, and they are enforced in
the same two places the workspace is — the hydrating join that turns ids into documents, and an
outbound re-check on the row in hand — and never in the vector store, which has no column for
either and will not get one.

Two values meet here.

**A reader set** is what a caller holds: a set of principals, each a string naming one person
(``user:<id>``) or one reader group (``group:<id>``). It travels on
:attr:`~manicule.core.retrieval.Filter.readers`, where ``None`` and ``frozenset()`` are opposite
instructions. ``None`` is a caller nothing restricts — the operator at this machine, or any
caller of an installation where no document carries a gate. An empty set is a caller who holds
no principal at all, and reads exactly the documents that carry no gate. Collapsing the two
would hand the second caller everything.

**A document's access** is the list of gates it must pass, every one of them. A gate is a set
of principals, any one of which opens it. A document with no gates is open to every reader of
the workspace, which is every document of an installation that configured nothing. A gate with
no principals opens for nobody: that is how a document whose readers could not be established is
recorded, so that "unknown" fails closed rather than reading as "unrestricted".

The gates are conjunctive because their origins are independent. An administrator restricting a
source to a group, and the source system restricting one page to its own readers, are two
decisions; a reader has to satisfy both, and neither can widen what the other allowed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

USER_PRINCIPAL: Final = "user:"
"""The prefix of a principal naming one person by their ``users.id``."""

GROUP_PRINCIPAL: Final = "group:"
"""The prefix of a principal naming one reader group by its id."""


def user_principal(user_id: str) -> str:
    """The principal for one person."""
    return f"{USER_PRINCIPAL}{user_id}"


def group_principal(group_id: str) -> str:
    """The principal for one reader group."""
    return f"{GROUP_PRINCIPAL}{group_id}"


def group_ids(readers: frozenset[str]) -> frozenset[str]:
    """The reader-group ids among ``readers``."""
    return frozenset(
        principal.removeprefix(GROUP_PRINCIPAL)
        for principal in readers
        if principal.startswith(GROUP_PRINCIPAL)
    )


class GateOrigin(StrEnum):
    """Who put a gate on a document."""

    SOURCE = "source"
    """An administrator granted the document's source to reader groups."""

    DOCUMENT = "document"
    """The source system restricted this document, and its connector recorded who may read it."""


class AccessGate(BaseModel):
    """One restriction a reader must pass: any of :attr:`readers` opens it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: GateOrigin
    readers: frozenset[str] = frozenset()
    """The principals that open this gate. Empty opens it for nobody."""


class DocumentAccess(BaseModel):
    """Every gate a document carries, as the store read them with the document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gates: tuple[AccessGate, ...] = ()

    @property
    def restricted(self) -> bool:
        """Whether any reader of the workspace could be refused this document."""
        return bool(self.gates)


class ReaderGroup(BaseModel):
    """A named set of people, and the sources an administrator granted to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    members: tuple[str, ...] = ()
    """The ``users.id`` of each person in it."""
    sources: tuple[str, ...] = ()
    """The sources granted to it."""


class ReadableTotals(BaseModel):
    """A workspace's document counts, over only the documents one reader set may read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: int = 0
    chunks: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_media_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


UNRESTRICTED: Final = DocumentAccess()
"""The access of a document that carries no gate."""


def admits(readers: frozenset[str] | None, access: DocumentAccess | None) -> bool:
    """Whether a caller holding ``readers`` may read a document with ``access``.

    ``access`` is ``None`` when the document was produced by a path that did not read its gates.
    That is not the same as having none, and a restricted caller is refused it: a re-check that
    passed whatever it could not see would be a re-check of nothing.
    """
    if readers is None:
        return True
    if access is None:
        return False
    return all(gate.readers & readers for gate in access.gates)


__all__ = [
    "GROUP_PRINCIPAL",
    "UNRESTRICTED",
    "USER_PRINCIPAL",
    "AccessGate",
    "DocumentAccess",
    "GateOrigin",
    "ReadableTotals",
    "ReaderGroup",
    "admits",
    "group_ids",
    "group_principal",
    "user_principal",
]
