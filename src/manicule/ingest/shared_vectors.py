"""Reuse of another workspace's stored vectors, without sharing a row with it.

``docs/storage.md`` §6.2.6 is the design this implements. The same source synced into two
workspaces is two documents with two sets of chunks, because a document's identity carries its
workspace (:func:`~manicule.core.ids.document_id`), and that is not negotiable. What it does not
have to be is two sets of forward passes: a vector is a pure function of its embedding input
under one fingerprint, so a vector the first workspace already holds for a chunk's exact
embedding input *is* the vector the model would return for the second workspace's chunk.

**How the match is made without loosening the identity.**
:func:`~manicule.core.embedding.embedding_input_identity` folds the document id into every
stored identity, so no value the second workspace can compute for its own chunk ever equals
one the first workspace stored. That is the tenancy boundary and it stays where it is. What a
lender does instead is ask each other workspace the question in *that* workspace's terms: the
document the same ``(source, source_id)`` would be there is
``document_id(other_workspace, source, source_id)``, so the chunk is re-addressed to that
document and the other workspace's own store is asked
:meth:`~manicule.core.protocols.VectorStore.stored_vectors` about it. Every part of the reuse
condition — the fingerprint, the embedding input, a readable vector whose recorded checksum
still describes it — is then decided by the rule that store already applies to itself
(:func:`~manicule.core.embedding.classify_stored_vector`), and nothing about it is written
twice. No store gains a method, so every backend, plugin ones included, lends through the
contract it already passes.

What comes back is a vector and only a vector. Nothing of the other workspace's rows — no
chunk, no id, no text — reaches this one: the borrowing workspace writes its own chunk under
its own id with its own identity, through its own store's ordinary
:meth:`~manicule.core.protocols.VectorStore.upsert`, exactly as it writes a model output.

**Not disclosed to anybody scoped to a workspace.** A caller of the borrowing workspace sees the
document as embedded (:class:`~manicule.ingest.embedding.EmbeddingWork`), and the count of
vectors borrowed is written only to an operator's ledger. What cannot be hidden is time: a
document that borrowed every vector indexes faster than one that did not, and with reuse always
on that difference is observable to anybody who can time a sync. That is an accepted cost,
stated in ``docs/storage.md`` §6.2.6 rather than discovered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from manicule.core.ids import document_id

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import EmbedFingerprint, Vector
    from manicule.core.protocols import VectorStore

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Donor:
    """Another workspace's vectors, opened for reading in the borrowing workspace's space."""

    workspace: str
    vectors: VectorStore


class Donors(Protocol):
    """The other workspaces worth asking, resolved when asked.

    Resolved per question rather than once, because a long-lived process outlives the set: a
    workspace indexed after this one's pipeline was built is a donor from then on, and one reset
    since is not one any more.
    """

    async def donors(self, *, source: str, source_id: str) -> Sequence[Donor]:
        """The other workspaces that may hold vectors for the document ``(source, source_id)``.

        A narrowing, never a guarantee: a donor returned here is still asked, and its store's
        own verdict decides whether anything is lent. Returning every other workspace is
        correct and merely slower.
        """
        ...


class BorrowLedger(Protocol):
    """Where the operator's count of borrowed vectors is kept."""

    async def record(self, vectors: int) -> None:
        """Add ``vectors`` copied from other workspaces instead of being embedded here."""
        ...


class SharedVectors:
    """Lends vectors other workspaces hold, one document at a time.

    Built once per pipeline and bound per document by :meth:`for_document`, because the match
    needs the document's ``(source, source_id)`` and a chunk carries only its document id.
    """

    def __init__(
        self,
        donors: Donors,
        *,
        fingerprint: EmbedFingerprint,
        ledger: BorrowLedger | None = None,
    ) -> None:
        """Lend from ``donors`` into a workspace that embeds with ``fingerprint``.

        ``fingerprint`` is the borrowing workspace's embedder, and the first clause of the reuse
        condition is checked against it here rather than trusted to whoever chose the donors.
        A donor store classifies its own rows against *its* recorded fingerprint, so a donor
        from another model would find its own rows perfectly readable — and lend a vector from
        a different space under a name that looks current.
        """
        self._donors = donors
        self._fingerprint = fingerprint
        self._ledger = ledger

    def for_document(self, source: str, source_id: str) -> DocumentLender:
        """A lender for the chunks of the document ``(source, source_id)`` names here."""
        return DocumentLender(self, source=source, source_id=source_id)

    async def lend(
        self, chunks: Sequence[Chunk], *, source: str, source_id: str
    ) -> Mapping[str, Vector]:
        """A readable stored vector, from some other workspace, for each chunk one exists for.

        Donors are asked in turn, each about only what the ones before it did not answer, and
        the first readable verdict for a chunk is the one kept. Which donor answers does not
        matter: every readable verdict is the stored output of one model on one input, which is
        what the fingerprint and the identity check establish.

        A donor that cannot be read is skipped rather than allowed to fail the document. Its
        vectors are an optimization the document was never entitled to; the model is still
        there.
        """
        found: dict[str, Vector] = {}
        remaining: list[Chunk] = list(chunks)
        if not remaining:
            return found
        for donor in await self._donors.donors(source=source, source_id=source_id):
            if not remaining:
                break
            try:
                recorded = await donor.vectors.fingerprint()
            except Exception as exc:  # noqa: BLE001 - a donor's failure is not this document's
                _LOG.warning(
                    "could not read another workspace's embedding fingerprint for reuse; "
                    "embedding instead: %s",
                    type(exc).__name__,
                )
                continue
            if recorded is None or not recorded.matches(self._fingerprint):
                continue
            addressed = document_id(donor.workspace, source, source_id)
            proxies: list[Chunk] = [
                chunk.model_copy(update={"document_id": addressed}) for chunk in remaining
            ]
            try:
                verdicts = await donor.vectors.stored_vectors(proxies)
            except Exception as exc:  # noqa: BLE001 - a donor's failure is not this document's
                _LOG.warning(
                    "could not read another workspace's vectors for reuse; embedding instead: %s",
                    type(exc).__name__,
                )
                continue
            for chunk in remaining:
                verdict = verdicts.get(chunk.id)
                if verdict is not None and verdict.is_reusable:
                    found[chunk.id] = verdict.vector
            remaining = [chunk for chunk in remaining if chunk.id not in found]
        if found and self._ledger is not None:
            try:
                await self._ledger.record(len(found))
            except Exception as exc:  # noqa: BLE001 - accounting must not fail an ingest
                _LOG.warning(
                    "could not record vectors reused across workspaces: %s", type(exc).__name__
                )
        return found


@dataclass(frozen=True, slots=True)
class DocumentLender:
    """:class:`SharedVectors` bound to one document, as :func:`embed_or_reuse` asks for it."""

    shared: SharedVectors
    source: str
    source_id: str

    async def lend(self, chunks: Sequence[Chunk]) -> Mapping[str, Vector]:
        """See :meth:`SharedVectors.lend`."""
        return await self.shared.lend(chunks, source=self.source, source_id=self.source_id)


__all__ = ["BorrowLedger", "DocumentLender", "Donor", "Donors", "SharedVectors"]
