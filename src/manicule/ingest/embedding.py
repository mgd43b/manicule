"""The one path by which chunks reach an embedder.

Every route that embeds stored chunks goes through :func:`embed_chunks` — first ingest,
``reindex --repair`` and ``reindex --re-embed`` alike. One function rather than three call
sites, because the check it performs is the sort that gets omitted from exactly one of them.

**Re-embed is the route this exists for.** It reads stored ``embed_text`` and does not
re-chunk, so the chunker's own budget refusal never runs; and
:attr:`~manicule.core.embedding.EmbedFingerprint.max_sequence_length` is excluded from
identity, so a limit that *fell* — a different checkpoint, an edited config, a changed backend
default — changes no fingerprint and fires no comparison. Every oversized chunk would then be
silently truncated into a vector claiming text it never saw, across a corpus, in one command,
with no error. :func:`~manicule.core.embedding.require_within_context` is what stands between
that and the index, and :func:`manicule.testing.assert_refuses_oversized_chunks` is what holds
this module to calling it.

**Embedding is serialised, and that is what an in-process embedder means.** With a model
server, concurrency is a connection-pool question and more requests mean more throughput. With
one model, one unified-memory pool and one accelerator, two concurrent batches produce
contention rather than throughput. So this is a single consumer, and the parallelism upstream
exists to keep it fed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.core.embedding import VectorState, require_within_context

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import Vector
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Embedder, VectorStore

MAX_BATCH = 64
"""Upper clamp on a derived batch size, so a tiny budget cannot ask for a huge batch."""


def batch_size(*, budget_tokens: int, target_batch_tokens: int, maximum: int = MAX_BATCH) -> int:
    """How many chunks to embed at once, derived rather than constant.

    A constant is wrong in both directions: thirty-two chunks of 512 tokens is a very
    different allocation from thirty-two of 8 000, and the second is where an in-process
    embedder runs the machine out of memory. Both inputs come from fingerprints already on
    hand, and the one tunable is ``target_batch_tokens`` — the honest place for the knob,
    because tokens are the quantity that maps to memory.
    """
    if budget_tokens <= 0:  # pragma: no cover - the fingerprint constrains this to be positive
        return 1
    return max(1, min(maximum, target_batch_tokens // budget_tokens))


async def embed_chunks(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    chunk_fingerprint: ChunkFingerprint | None = None,
    target_batch_tokens: int = 16_384,
    maximum: int = MAX_BATCH,
    on_batch: Callable[[Sequence[Chunk]], None] | None = None,
) -> list[Vector]:
    """Embed ``chunks`` in order, refusing any the model would truncate.

    Args:
        embedder: The configured embedder. Called one batch at a time, never concurrently.
        chunks: What to embed. ``embed_text`` is what goes to the model; ``text`` is never
            sent, because it carries no breadcrumb and a section called "Configuration" is
            unretrievable without knowing what it configures.
        chunk_fingerprint: The chunker that produced them, when it is known. Supplying it adds
            a tokenizer check, because a token count taken under a different vocabulary is not
            a measurement of anything relevant to this model's limit.
        target_batch_tokens: Tokens per batch, from which the batch size is derived.
        maximum: Clamp on the derived size.
        on_batch: Called with each batch immediately before the model sees it. This is the
            only place in the process that knows how many forward passes an operation cost,
            and a count derived afterwards from ``len(chunks) // size`` would be a restatement
            of the arithmetic above rather than a measurement of what happened.

    Returns:
        One vector per chunk, in the order the chunks were given.

    Raises:
        ContextOverflowError: Any chunk exceeds what the model will read.
    """
    require_within_context(chunks, embedder.fingerprint, chunk_fingerprint)
    if not chunks:
        return []

    budget = (
        chunk_fingerprint.max_tokens
        if chunk_fingerprint
        else embedder.fingerprint.max_sequence_length
    )
    size = batch_size(
        budget_tokens=budget, target_batch_tokens=target_batch_tokens, maximum=maximum
    )

    vectors: list[Vector] = []
    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        if on_batch is not None:
            on_batch(batch)
        produced = await embedder.embed([chunk.embed_text for chunk in batch])
        if len(produced) != len(batch):
            msg = (
                f"the embedder returned {len(produced)} vector(s) for {len(batch)} chunk(s). "
                f"They are positional, so a mismatch would store some chunk against another "
                f"chunk's vector — every citation from that batch onward would point somewhere "
                f"it does not belong."
            )
            raise ValueError(msg)
        vectors.extend(produced)
    return vectors


@dataclass(frozen=True)
class EmbeddingWork:
    """What one call to :func:`embed_or_reuse` cost, in the terms that are not each other.

    Every field here exists because collapsing it into another one produces a report that
    claims something it did not check. A chunk keeping its id is not a vector surviving; a
    vector surviving is not a forward pass avoided; a forward pass avoided by this partition
    is not one absorbed by the embedder's in-memory cache. Those are four different facts and
    an operator pricing a corpus-wide re-parse needs them apart.

    Two invariants hold by construction, and the tests assert them rather than trusting them:

    - ``reused + embedded == chunks``
    - ``input_changed + repaired == embedded == vectors_new + vectors_replaced``
    """

    chunks: int = 0
    """Chunks the operation was given."""

    reused: int = 0
    """Vectors taken from the store without a model call, because the embedding input matched."""

    embedded: int = 0
    """Chunks handed to the embedder. The honest count of embedding work done."""

    input_changed: int = 0
    """Of ``embedded``, those whose embedding input is new or has changed.

    Includes the chunk whose ``text`` — and therefore whose id — did not move at all while the
    heading breadcrumb in its ``embed_text`` did. That chunk is why reuse is not keyed on the
    chunk id, and it is counted here rather than under ``reused``.
    """

    repaired: int = 0
    """Of ``embedded``, those whose embedding input was unchanged and whose vector was not usable.

    A row that went missing, or one whose stored vector cannot be read at the index's
    dimension, or one whose recorded identity contradicts the chunk stored beside it. Identity
    metadata asserting that a vector exists is not the same as the vector existing, so this
    group is found by looking rather than by trusting.
    """

    forward_calls: int = 0
    """Batches the embedder was actually asked for. Counted at the call, not derived from it."""

    vectors_new: int = 0
    """Rows written where the store held none."""

    vectors_replaced: int = 0
    """Rows whose stored vector was overwritten with a different one.

    A reused vector is written back unchanged — the row's chunk may have moved position, and
    an unrecorded identity is recorded by the write — so it is not counted here. "Replaced"
    means the vector changed.
    """

    vectors_backfilled: int = 0
    """Reused rows that carried no recorded identity and had it reconstructed.

    The one-time migration, counted as it happens. It falls to zero once every row a sweep
    touches has been written since the identity column arrived; it is not a cost, because a
    reconstructed identity avoids the same forward pass a recorded one does.
    """


async def embed_or_reuse(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    vectors: VectorStore,
    chunk_fingerprint: ChunkFingerprint | None = None,
    previous: Mapping[str, str] | None = None,
    target_batch_tokens: int = 16_384,
    maximum: int = MAX_BATCH,
) -> tuple[list[Vector], EmbeddingWork]:
    """Embed only the chunks whose embedding input the index does not already hold a vector for.

    **The reuse condition is three-part and every part is load-bearing**: the same embedding
    fingerprint, the same embedding input, *and* a readable stored vector for that identity.
    The first two are answered by
    :func:`~manicule.core.embedding.embedding_input_identity`; the third is answered by
    reading the row, because identity metadata claiming a vector exists is not a vector
    existing. Anything weaker keeps a stale vector under current chunk text — in particular
    reuse keyed on the chunk id, which survives a re-parse whenever ``text`` does while the
    breadcrumb in ``embed_text`` may not have.

    **The oversize refusal runs over every chunk before anything is reused.** A document all of
    whose vectors are reusable would otherwise never reach
    :func:`~manicule.core.embedding.require_within_context`, and a corpus-wide sweep would
    then pass in silence under a model whose sequence length had *fallen* — the failure the
    module docstring is about, which changes no fingerprint and fires no comparison.

    Args:
        embedder: The configured embedder, called only for what is not reusable.
        chunks: The complete, ordered chunk set about to be committed.
        vectors: The store to ask, and the store the answer is about. It is asked about chunks
            rather than ids, so that its answer can be about the embedding input.
        chunk_fingerprint: The chunker that produced ``chunks``, when it is known.
        previous: Chunk id to the ``embed_text`` the index already held for it, when the caller
            knows. It separates two chunks that both have no vector row: one that never had one
            because it is new, and one whose row went missing while its input did not change.
            Both are embedded either way — this only decides which of them the report calls a
            repair. An absent mapping is not a claim that nothing was stored; it is a caller
            that did not look, and every chunk with no row is then counted as new.
        target_batch_tokens: Tokens per batch, from which the batch size is derived.
        maximum: Clamp on the derived size.

    Returns:
        One vector per chunk, in the order the chunks were given, and what it cost.

    Raises:
        ContextOverflowError: Any chunk exceeds what the model will read.
    """
    require_within_context(chunks, embedder.fingerprint, chunk_fingerprint)
    if not chunks:
        return [], EmbeddingWork()

    verdicts = await vectors.stored_vectors(chunks)
    held = previous or {}

    reused: dict[int, Vector] = {}
    pending: list[Chunk] = []
    positions: list[int] = []
    counts = dict.fromkeys(VectorState, 0)
    backfilled = 0
    repaired = 0

    for position, chunk in enumerate(chunks):
        verdict = verdicts[chunk.id]
        counts[verdict.state] += 1
        if verdict.is_reusable:
            reused[position] = verdict.vector
            backfilled += 0 if verdict.identity_recorded else 1
            continue
        if verdict.state is VectorState.CORRUPT or (
            verdict.state is VectorState.ABSENT and held.get(chunk.id) == chunk.embed_text
        ):
            repaired += 1
        pending.append(chunk)
        positions.append(position)

    forward_calls = 0

    def count_batch(batch: Sequence[Chunk]) -> None:
        nonlocal forward_calls
        del batch
        forward_calls += 1

    produced = await embed_chunks(
        embedder,
        pending,
        chunk_fingerprint=chunk_fingerprint,
        target_batch_tokens=target_batch_tokens,
        maximum=maximum,
        on_batch=count_batch,
    )

    ordered: list[Vector] = [()] * len(chunks)
    for position, vector in reused.items():
        ordered[position] = vector
    for position, vector in zip(positions, produced, strict=True):
        ordered[position] = vector

    return ordered, EmbeddingWork(
        chunks=len(chunks),
        reused=len(reused),
        embedded=len(pending),
        input_changed=len(pending) - repaired,
        repaired=repaired,
        forward_calls=forward_calls,
        vectors_new=counts[VectorState.ABSENT],
        vectors_replaced=counts[VectorState.STALE] + counts[VectorState.CORRUPT],
        vectors_backfilled=backfilled,
    )


__all__ = ["MAX_BATCH", "EmbeddingWork", "batch_size", "embed_chunks", "embed_or_reuse"]
