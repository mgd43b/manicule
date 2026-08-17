"""The four checks that run once per run, before the first document is discovered.

Two of them are specified elsewhere and merely *run* here; the other two exist only here,
because nothing else holds both halves of them.

0. **The boundaries were measured, not estimated** — ``docs/parsing.md`` §1.2. First, because
   a corpus chunked with a stand-in vocabulary cannot be made admissible by anything the
   three checks below discover, and because it is the one refusal that is about the running
   configuration alone and needs to read nothing.

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
        PolicyError: The chunk budget exceeds what the model will read, or the boundaries were
            measured with a stand-in vocabulary.
    """
    require_measured(chunk)
    require_coherent(embed=embed, chunk=chunk)

    stored = await store.index_fingerprints()
    if stored.embed is not None:
        await _refuse_embed_mismatch(stored.embed, embed, store)
    if stored.chunk is not None:
        await _refuse_chunk_mismatch(stored.chunk, chunk, store)

    if vectors is not None:
        held = await vectors.fingerprint()
        if held is not None:
            await _refuse_embed_mismatch(held, embed, store)

    committed = IndexFingerprints(
        embed=embed, chunk=chunk, vector_table=_vector_table(embed, vectors)
    )
    if stored != committed:
        await store.record_index_fingerprints(committed)
    return committed


def _vector_table(embed: EmbedFingerprint, vectors: VectorStore | None) -> str | None:
    """Which table this index's vectors are in, for ``index_state.vector_table``.

    ``docs/storage.md`` §6.5 says a first ingest "creates ``chunks__<fp8>`` … and sets
    ``index_state.vector_table`` in the same SQLite transaction that records the fingerprint".
    Nothing did. The caller carried ``stored.vector_table`` forward, which is ``NULL`` on a
    first ingest and therefore ``NULL`` for ever: the column shipped, the backup manifest
    carried it, the retrieval trace reported it, and ``doctor`` printed a healthy index as
    "13 document(s) in no vector table" — with no code path anywhere that had ever written a
    value into it.

    A publication-aware store reports the SQLite pointer it just resolved, because after a
    shadow swap that pointer names a generation directory rather than the inner Lance table.
    A plain store is derived through the same ``table_name(fingerprint)`` function it uses.

    Returns:
        The table name, or ``None`` when there is no vector store — because then there is no
        table, and recording a name for one would be describing something that does not exist.
        The live generation pointer is preferred when the store exposes one; replacing it with
        the inner table name would silently roll a successful re-embedding back to the legacy
        directory during the next ingest refusal check.
    """
    if vectors is None:
        return None
    publication_pointer = getattr(vectors, "publication_pointer", None)
    if isinstance(publication_pointer, str):
        return publication_pointer
    from manicule.storage.vectors import table_name  # noqa: PLC0415 - a storage extra

    return table_name(embed)


def require_measured(chunk: ChunkFingerprint) -> None:
    """Refuse boundaries taken with a stand-in vocabulary rather than the model's own.

    ``docs/parsing.md`` §1.2 is titled "Count with the embedder's tokenizer, never an
    estimator", and ends by saying provisional chunks never reach the index. This is that
    sentence, in code. It had been a docstring in three modules and a refusal in none of them,
    which is the shape ``docs/contracts.md`` §5 calls worse than an absent guarantee: the
    stand-in inflates every count by a factor chosen without measuring anything, so the
    boundaries are neither the model's nor reproducible from it, and every downstream check
    was written on the assumption that a token count means what the model means by it.

    **A refusal rather than a better identifier, and the identifier as well.** Making the id
    honest (``provisional:x1.5:tiktoken/cl100k_base@0.13.0`` rather than ``provisional``) is
    what stops two estimated corpora being mistaken for one another; it does not make either
    of them fit to serve. The id is what this check reads, so the two halves are one
    mechanism rather than two guards that can drift apart.

    Raises:
        PolicyError: The chunker counted without a bound embedder.
    """
    if not chunk.provisional:
        return
    msg = (
        f"these chunk boundaries were measured with {chunk.tokenizer_id!r}, which is a "
        f"stand-in for the embedder's tokenizer rather than the tokenizer itself. The count "
        f"is inflated by a fixed safety factor and can still undercount by an unknown margin "
        f"under a vocabulary the model does not use, and undercounting is the direction that "
        f"truncates without raising. Provisional chunks are for inspection — a dry-run parse, "
        f"a fixture build — and never for an index. Bind an embedder and chunk again."
    )
    raise PolicyError(msg)


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


async def _refuse_chunk_mismatch(
    stored: ChunkFingerprint, offered: ChunkFingerprint, store: IngestStore
) -> None:
    """Compare chunk structure and name the retained-source replacement workflow."""
    if stored.matches(offered):
        return
    chunks = await store.count_chunks()
    try:
        stored.require_match(offered)
    except FingerprintMismatchError as exc:
        note = (
            f"{chunks} stored chunk(s) must be rechunked from retained source snapshots; "
            "re-embedding stored embed_text cannot change structural boundaries. Run "
            "`manicule rebuild plan SNAPSHOT_ID` for an aggregate cost/capacity estimate, "
            "then `manicule rebuild execute SNAPSHOT_ID` to publish one resumable atomic "
            "workspace generation without reacquiring source data."
        )
        exc.add_note(note)
        raise


__all__ = ["check_before_run", "require_coherent", "require_measured"]
