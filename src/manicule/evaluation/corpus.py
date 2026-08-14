"""Which corpus produced a result, pinned so that results stay comparable as it grows.

The property this exists to protect is stated once and everything below follows from it:

    **Same content on both sides, so a difference is retrieval and not what was indexed.**

Two ways that quietly stops being true, and each needs a different instrument.

**Across the two sides of one comparison.** The label is an operator's assertion that both
systems are pointed at the same documents, and it is checked before any preference is
recorded. It is an assertion rather than a proof because one side may be a system manicule
cannot introspect at all — an adapter and a configuration label supplied at runtime is the
whole of what is required of it.

**Across runs of the same side, weeks apart.** Here the label is the *weakest* instrument,
because it is the thing that does not change when the corpus does: documents get added, the
label still says ``knowledge-base``, and last month's win rate is compared against this
month's over different content. So a system that can compute one records a digest of what it
holds, and a report over records whose digests differ refuses rather than averaging.

Absence of a digest is recorded as absence, never as agreement. A report where either side
could not produce one says the corpus identity was asserted rather than verified, on the face
of the report and not in a footnote.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.retrieval import Filter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from manicule.core.protocols import DocStore


@runtime_checkable
class SupportsChunkCount(Protocol):
    """A store that can say how many chunks it holds.

    Declared here and narrower than anything in ``manicule.retrieval.ports``, because that is
    all this module needs: a store that cannot count chunks is an ordinary store, and the
    corpus version simply records the count as absent rather than as zero.
    """

    async def count_chunks(self, document_id: str | None = None) -> int: ...


DIGEST_PAGE = 500
"""Documents read per round trip when computing a digest.

Large enough that a ten-thousand-document corpus is twenty statements, small enough that no
single query materializes a corpus-sized result. The digest is computed once per evaluation
session, not per query.
"""


class CorpusVersion(BaseModel):
    """What was indexed when a result was produced.

    Travels on every result and into every recorded preference, so a stored judgment can be
    read back years later and still say what it was a judgment about.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(
        min_length=1,
        description="The operator's name for this content. Required on both sides and checked "
        "before anything is recorded: comparing two systems over different documents measures "
        "the documents.",
    )
    digest: str | None = Field(
        default=None,
        description="A content-derived identity, where the system can compute one. ``None`` "
        "means the corpus identity rests on the label alone — recorded as such, and surfaced "
        "in the report, because an unverifiable claim presented as a verified one is the "
        "failure this whole type exists to prevent.",
    )
    document_count: int | None = Field(
        default=None,
        ge=0,
        description="Live documents a search chooses between. The probe needs this to state "
        "what chance looks like, and refuses to produce a verdict without it.",
    )
    chunk_count: int | None = Field(default=None, ge=0)
    embed_fingerprint: str | None = Field(
        default=None,
        description="The vector space this side embedded into, where it has one. Two runs in "
        "different spaces are not two measurements of the same index.",
    )

    def disagreement_with(self, other: CorpusVersion) -> str | None:
        """Why these two may not be compared, or ``None`` if nothing rules it out.

        Deliberately narrow. Only two things are refusals: different labels, and two digests
        that disagree. Counts are *recorded* and reported but never refused on, because two
        systems chunk differently and a chunk-count mismatch between them is ordinary rather
        than evidence of different content — a refusal there would block exactly the
        cross-system comparison this is built for, for a reason that is not about the corpus.
        """
        if self.label != other.label:
            return (
                f"corpus labels differ: {self.label!r} against {other.label!r}. Both sides must "
                f"be pointed at the same documents, or the comparison measures the documents."
            )
        if self.digest is not None and other.digest is not None and self.digest != other.digest:
            return (
                f"corpus label {self.label!r} is shared but the digests differ "
                f"({self.digest} against {other.digest}), so the label is stale on one side."
            )
        return None

    def agrees_verifiably_with(self, other: CorpusVersion) -> bool:
        """Whether sameness was *checked* rather than asserted.

        ``False`` whenever either side could not produce a digest. The report prints this, so
        "we compared the same corpus" is never stronger in the summary than it was in the run.
        """
        return self.digest is not None and other.digest is not None and self.digest == other.digest


def digest_of(entries: Iterable[tuple[str, str]]) -> str:
    """A stable digest over ``(document id, content hash)`` pairs.

    Sorted before hashing, so two stores that enumerate in different orders agree. Content
    hashes rather than document ids alone, because a corpus whose documents were all edited has
    the same ids and is not the same corpus.
    """
    hasher = hashlib.sha256()
    for document_id, content_hash in sorted(entries):
        hasher.update(document_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(content_hash.encode("utf-8"))
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


async def corpus_version_of(
    docstore: DocStore,
    *,
    label: str,
    workspace_ids: frozenset[str],
    embed_fingerprint: str | None = None,
) -> CorpusVersion:
    """Read a store and describe what it holds.

    Enumerates rather than trusting a counter: the digest is the point, and a count without one
    cannot tell a corpus that grew from one that was replaced.

    Args:
        docstore: The store to describe.
        label: The operator's name for this content, shared with whatever it is compared to.
        workspace_ids: The scope the evaluation runs in. Its own argument rather than a whole
            :class:`~manicule.core.retrieval.Filter`, because a corpus version restricted by
            source or tag would describe a slice while claiming to describe the corpus.
        embed_fingerprint: The vector space, where the caller knows it.
    """
    scope = Filter(workspace_ids=workspace_ids)
    entries: list[tuple[str, str]] = []
    offset = 0
    while True:
        page = await docstore.list_documents(scope, limit=DIGEST_PAGE, offset=offset)
        if not page:
            break
        entries.extend((document.id, document.content_hash) for document in page)
        if len(page) < DIGEST_PAGE:
            break
        offset += DIGEST_PAGE

    chunk_count: int | None = None
    if isinstance(docstore, SupportsChunkCount):
        chunk_count = await docstore.count_chunks()

    return CorpusVersion(
        label=label,
        digest=digest_of(entries),
        document_count=len(entries),
        chunk_count=chunk_count,
        embed_fingerprint=embed_fingerprint,
    )


__all__ = [
    "DIGEST_PAGE",
    "CorpusVersion",
    "SupportsChunkCount",
    "corpus_version_of",
    "digest_of",
]
