"""The three checks that run once per run, before the first document is discovered.

Two of them are specified elsewhere and merely *run* here; the third exists only here,
because nothing else holds both halves of it.

1. ``EmbedFingerprint`` — configuration against ``index_state`` against the vector store's own
   record of what it holds (``docs/storage.md`` §6.3). Three places, because two cannot detect
   the interesting failure: swapping a ``vectors/`` directory for another instance's, or
   restoring half a backup.
2. ``ChunkFingerprint`` — configuration against ``index_state``, **with the middleware
   declaration set folded in** (``docs/ingest.md`` §3.3). Without the fold, two instances with
   identical configuration and different middleware produce different vectors from identical
   source bytes and neither refusal notices.
3. ``max_tokens <= max_sequence_length`` — a cross-check between the two, not a property of
   either. Each is individually valid while the pair is incoherent: a 512-token budget against
   a 256-token model silently truncates every chunk, producing vectors for the first half of
   the text and citations that point at all of it.

**Once per run, and before discovery.** Per-document would run the check tens of thousands of
times to answer a question whose inputs cannot change mid-run — and, worse, a corpus that
happens to be entirely unchanged skips every document, so a mismatched configuration would go
undiscovered until the first new document arrived. Before discovery rather than after, because
discovery is the rate-limited part and refusing after a forty-minute enumeration is a worse
version of refusing immediately.

Every refusal names its price. ``--re-embed`` reads stored ``embed_text`` and touches neither
the network nor a parser, so the operator is choosing between a configuration change and a
known number of chunks — not between a configuration change and an open-ended one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.embedding import IndexFingerprints
from manicule.core.errors import FingerprintMismatchError, PolicyError

if TYPE_CHECKING:
    from manicule.core.embedding import EmbedFingerprint
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import VectorStore
    from manicule.ingest.ports import IngestStore


async def check_before_run(
    *,
    embed: EmbedFingerprint,
    chunk: ChunkFingerprint,
    store: IngestStore,
    vectors: VectorStore | None = None,
) -> IndexFingerprints:
    """Refuse to start unless this configuration can write to this index.

    Args:
        embed: What the configured embedder produces.
        chunk: What the configured chunker produces, **already carrying its middleware
            declarations** — see
            :meth:`~manicule.core.fingerprints.ChunkFingerprint.with_middleware`. Passing the
            chunker's bare fingerprint is the mistake this argument's name exists to prevent.
        store: Where ``index_state`` lives.
        vectors: The vector store, when there is one. Its own record is the third comparison,
            and the only one that notices a swapped directory.

    Returns:
        The state the index is now committed to, so a caller does not read it twice.

    Raises:
        FingerprintMismatchError: The index was built by something else.
        PolicyError: The chunk budget exceeds what the model will read.
    """
    require_coherent(embed=embed, chunk=chunk)

    stored = await store.index_fingerprints()
    if stored.embed is not None:
        await _refuse_embed_mismatch(stored.embed, embed, store)
    if stored.chunk is not None:
        stored.chunk.require_match(chunk)

    if vectors is not None:
        held = await vectors.fingerprint()
        if held is not None:
            await _refuse_embed_mismatch(held, embed, store)

    committed = IndexFingerprints(embed=embed, chunk=chunk, vector_table=stored.vector_table)
    if stored != committed:
        await store.record_index_fingerprints(committed)
    return committed


def require_coherent(*, embed: EmbedFingerprint, chunk: ChunkFingerprint) -> None:
    """Refuse a chunk budget the embedder will not read to the end of.

    The single reason this is a startup refusal rather than a per-document guard. An
    in-process embedder gets none of the protection the parse stage gets — it cannot be moved
    behind a process boundary without either reloading the model per worker or reintroducing
    the server the design rejects — so the honest response is to remove the failure mode
    rather than to catch it. A too-long input cannot reach the embedder because a
    configuration that would produce one does not start.

    Raises:
        PolicyError: The budget exceeds the model's effective context.
    """
    if chunk.max_tokens <= embed.max_sequence_length:
        return
    msg = (
        f"the chunk budget is {chunk.max_tokens} tokens but {embed.describe()} attends to "
        f"{embed.max_sequence_length}. Each setting is valid on its own and the pair is not: "
        f"every chunk past the limit would be truncated with no error raised, storing a vector "
        f"that describes an opening fragment while the chunk still claims all of its text — a "
        f"citation quoting words the index never saw. Lower the chunker's max_tokens to "
        f"{embed.max_sequence_length} or fewer, or configure a model with a longer context."
    )
    raise PolicyError(msg)


async def _refuse_embed_mismatch(
    stored: EmbedFingerprint, offered: EmbedFingerprint, store: IngestStore
) -> None:
    """Compare, and price the repair when they differ."""
    if stored.matches(offered):
        return
    chunks = await store.count_chunks()
    try:
        stored.require_match(offered)
    except FingerprintMismatchError as exc:
        note = (
            f"{chunks} stored chunk(s) would need re-embedding. `reindex --re-embed` reads "
            f"chunks.embed_text and touches neither the network nor a parser."
        )
        exc.add_note(note)
        raise


__all__ = ["check_before_run", "require_coherent"]
