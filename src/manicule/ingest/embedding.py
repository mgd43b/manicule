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

from typing import TYPE_CHECKING

from manicule.core.embedding import require_within_context

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk
    from manicule.core.embedding import Vector
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Embedder

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


__all__ = ["MAX_BATCH", "batch_size", "embed_chunks"]
