"""Embedding vocabulary: vectors, token states, and the fingerprint that identifies them.

**Nothing in manicule may hardcode a vector dimension.** The dimension is read from
:class:`EmbedFingerprint` at run time, the vector table is created at first ingest with
whatever the embedder reports, and ingest refuses to start when the fingerprint does not
match the one the index was built with.

Dimension alone is not identity. Two unrelated 1024-dimension models produce vectors that
are mutually meaningless, and a guard that compared only ``D`` would wave that through
while quietly destroying retrieval quality. :meth:`EmbedFingerprint.identity` is what gets
compared.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import ClassVar, Final, Protocol, override, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.core.content import Chunk
from manicule.core.errors import ContextOverflowError
from manicule.core.fingerprints import ChunkFingerprint, Fingerprint

type Vector = Sequence[float]
"""A finished embedding.

Deliberately a plain sequence: vectors cross a storage boundary, where a concrete,
serializable value is what is wanted. Backends holding native arrays convert at the seam.
"""


def is_finite_vector(vector: Vector) -> bool:
    """Whether every component can participate in distance arithmetic.

    ``NaN`` and infinities have the right shape and serialize cleanly, but cosine distance
    against them is undefined. Treating one as readable would preserve a transient numerical
    failure as durable retrieval state.
    """
    return all(math.isfinite(value) for value in vector)


@runtime_checkable
class NDArrayLike(Protocol):
    """The little that core needs to know about a numeric array.

    Token states are large and stay in whatever array type the backend produced them in —
    core never touches the values, so it never needs the library that owns them. Both numpy
    arrays and MLX arrays satisfy this.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[object]: ...


class Pooling(StrEnum):
    """How token states are reduced to one vector per text.

    This is part of a model's identity, not a preference. On a typical retrieval model,
    CLS and mean pooling of the same token states differ by around 0.86 cosine: both look
    like plausible vectors, one retrieves materially worse, and no error is raised. Mixing
    them within an index is therefore a correctness failure, which is why changing the
    value changes the fingerprint.
    """

    CLS = "cls"
    MEAN = "mean"
    LAST_TOKEN = "last_token"  # noqa: S105 - a pooling strategy, not a credential
    NONE = "none"


class TokenStates(BaseModel):
    """Pre-pooled per-token hidden states for a batch of texts — the tier A payload.

    manicule pools these itself. Backends have been observed to bind a
    ``last_hidden_state``-shaped attribute to the *pooled* vector, one attribute above the
    real token states, so trusting the field name silently yields the provider's pooling
    choice instead of the configured one.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    states: NDArrayLike
    """Shape ``(batch, sequence, dimension)``."""

    attention_mask: NDArrayLike
    """Shape ``(batch, sequence)``. Required: mean pooling over padding is wrong."""

    dimension: int = Field(gt=0)


class EmbedFingerprint(Fingerprint):
    """The identity of the thing that produced a set of vectors.

    Persisted alongside an index and compared before every write.

    Three fields are recorded and excluded from identity. Every exclusion is deliberate, and
    each rests on something that has to stay true.

    **:attr:`max_sequence_length` is excluded because the invariant it protects is checked
    directly.** Including it would force a full re-embed whenever the limit *rises*, which
    changes nothing about the vectors already stored. What matters is not whether the number
    changed but whether any text was truncated, and :func:`require_within_context` answers
    that question about the actual batch. Every path that embeds stored chunks must call it
    — in particular the re-embed path, which reads stored ``embed_text`` without re-chunking
    and therefore never runs the chunker's own budget check. That is the one route by which
    a lowered limit could otherwise truncate a whole corpus in silence.

    **:attr:`backend` is excluded only because :attr:`weights_identity` carries the runtime
    boundary.** Explicitly pinned built-in artifact pairs that have passed the parity suite
    share one identity; every other artifact identity includes its backend. Portability is
    therefore an allowlisted measurement, not an inference from a model name.

    **:attr:`weights_ref` is provenance, while :attr:`weights_identity` is compatibility.**
    The former records the exact repository commit or local digest that executed. The latter
    is equal across backends only for a pinned pair that the parity suite qualifies.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_id",
        "revision",
        "dimension",
        "pooling",
        "normalized",
        "tokenizer_id",
        "weights_identity",
    )

    model_id: str = Field(min_length=1, description="Model identifier, e.g. ``BAAI/bge-m3``.")
    revision: str | None = Field(
        default=None,
        description="Model revision or commit. ``None`` means unpinned, which is recorded "
        "faithfully rather than filled in with a guess.",
    )
    dimension: int = Field(gt=0, description="Read from the model. Never a literal.")
    pooling: Pooling
    normalized: bool = Field(description="Whether vectors are L2-normalized on output.")

    tokenizer_id: str = Field(
        default="",
        description="Which tokenizer this model uses. Part of identity because token counts "
        "are tokenizer-specific, so a chunk budget agreed under one vocabulary means "
        "something else under another.",
    )
    max_sequence_length: int = Field(
        gt=0,
        description="The most tokens this model will actually attend to. **Required**, and "
        "the value every batch is checked against by ``require_within_context``: beyond it "
        "the input is truncated with no error raised, so an over-budget chunk is indexed as "
        "its opening tokens while still claiming all of its text — a citation quoting words "
        "the index never saw. "
        "Note this is the *effective* limit, which is not always the positional-embedding "
        "limit: some widely used models ship configured well below what their architecture "
        "allows, so it must be read from the loaded model rather than assumed. "
        "Excluded from identity — see the class docstring for why that is safe.",
    )

    backend: str = Field(
        default="",
        description="Which runtime produced the vectors, e.g. ``mlx`` or ``onnx``. "
        "Excluded from identity — see the class docstring, which records what that "
        "exclusion is betting on.",
    )
    weights_ref: str = Field(
        default="",
        description="The artifact whose bytes the backend actually executed, when that is "
        "not the model's own repository — for example ``mlx-community/bge-m3-mlx-fp16`` for "
        "``BAAI/bge-m3`` under MLX, which publishes no safetensors. Recorded so that a "
        "vector can be traced to the weights that made it. Excluded from identity because "
        "weights_identity expresses compatibility.",
    )
    weights_identity: str = Field(
        default="",
        description="Stable identity of the executable artifact. Equal across backends only "
        "for an explicitly pinned and parity-qualified built-in pair.",
    )

    @override
    def identity(self) -> dict[str, JsonValue]:
        """Compatibility fields, omitting the marker absent from legacy/plugin fingerprints."""
        identity = super().identity()
        if not self.weights_identity:
            identity.pop("weights_identity")
        return identity

    @override
    def describe(self) -> str:
        """A one-line human-readable form, for error messages and diagnostics."""
        revision = f"@{self.revision}" if self.revision else ""
        norm = "normalized" if self.normalized else "unnormalized"
        return f"{self.model_id}{revision} ({self.dimension}d, {self.pooling.value}, {norm})"


class IndexFingerprints(BaseModel):
    """What an index says it was built with.

    One row, read before a run and written when a run commits to a shape. Every field is
    optional because an empty index has committed to nothing yet, and ``None`` is the honest
    answer — it accepts whatever the first ingest brings, which is the one moment at which no
    comparison is possible and none is needed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embed: EmbedFingerprint | None = None
    chunk: ChunkFingerprint | None = None
    vector_table: str | None = Field(
        default=None,
        description="Which vector table holds the vectors. A pointer rather than a constant, "
        "because a re-embed builds its replacement alongside the live one and there is a "
        "window in which the index is neither the old thing nor the new one.",
    )

    @property
    def is_empty(self) -> bool:
        """Whether this index has committed to nothing, so anything is acceptable."""
        return self.embed is None and self.chunk is None


EMBEDDING_IDENTITY_VERSION: Final = "1"
"""What :func:`embedding_input_identity` hashes first, so its own rules can change.

A stored identity is compared byte for byte against a freshly derived one. If the derivation
ever changes — a different digest, a different serialization, a fourth input — every stored
identity has to stop matching, or vectors produced under the old rule would be reused under
the new one. Bumping this is what makes that happen, and it costs exactly one re-embed of the
corpus, which is the honest price of changing what the identity means.
"""


def embedding_input_identity(
    embed_text: str,
    *,
    document_id: str,
    embed: EmbedFingerprint,
    middleware: Sequence[str] = (),
) -> str:
    """The identity of one embedding *input*, which is not the identity of a chunk.

    A chunk's id is derived from its :attr:`~manicule.core.content.Chunk.text`; what reaches
    the model is :attr:`~manicule.core.content.Chunk.embed_text`, which carries the heading
    breadcrumb. So a chunk can keep its id while the string that produced its vector changes,
    and a reuse rule keyed on the id alone preserves a stale vector under current text. This
    is the value that rule has to be keyed on instead.

    Four inputs, and each one is load-bearing:

    ``document_id``
        **The tenancy boundary, put inside the identity rather than applied afterwards** —
        exactly as :func:`~manicule.core.ids.document_id` puts ``workspace_id`` inside a
        document's, and for the same reason. Reuse is answered by looking a stored identity up
        in a vector table that has no workspace column and by design never will
        (``docs/storage.md`` §6.2: tenancy lives on ``documents``, and a copy in a derived
        store is a value that can disagree). A lookup keyed on the embedding input alone is
        therefore a read no filter scopes, and "the caller will remember to scope it" is the
        assumption this repository has already had to fix once. Because a document id is
        derived from the workspace, folding it in makes a cross-tenant match impossible to
        express rather than merely unlikely to be written.

        Nothing measurable is lost. What it forgoes is reuse *between* two documents, and a
        chunk's ``embed_text`` carries the document's own title in its breadcrumb (§5.1), so
        two documents almost never produce the same embedding input in the first place. What it
        keeps is the case reuse exists for: the same document re-parsed, where a chunk id moves
        because it carries its position and the embedded string does not move at all.

    ``embed_text``
        The exact string handed to the embedder — every code point of it, in order. There is
        deliberately **no Unicode normalization**: NFC and NFD forms of one word tokenize
        differently and produce different vectors, so treating them as one input would reuse a
        vector for text the model never saw. "Normalized" here means the *serialization* is
        canonical, never that the text is.

    ``embed``
        The model. :meth:`~manicule.core.fingerprints.Fingerprint.canonical` rather than the
        model id, so two checkpoints of one name are two identities and the vectors of one are
        never reused for the other.

    ``middleware``
        ``name@version`` for every middleware declaring ``mutates_embedded_text``
        (:attr:`~manicule.core.fingerprints.ChunkFingerprint.embed_text_middleware`). Sorted
        here rather than trusted to arrive sorted, because a set that differs only in order is
        the same declaration. Folding it in is defense in depth over
        :func:`~manicule.ingest.refusals.check_before_run`, which refuses a run whose chunk
        fingerprint disagrees with the index — that refusal is per *run*, and the repair verbs
        in :mod:`manicule.ingest.reindex` do not all go through it.

    Args:
        embed_text: What was, or is about to be, sent to the model.
        document_id: The document the chunk belongs to, which carries its workspace.
        embed: The embedder's fingerprint.
        middleware: Declarations of middleware that may rewrite embedded text.

    Returns:
        A hex SHA-256 digest. Opaque, fixed width, and safe to store in a text column.
    """
    payload = json.dumps(
        [
            EMBEDDING_IDENTITY_VERSION,
            document_id,
            embed.canonical(),
            sorted(middleware),
            embed_text,
        ],
        # A JSON array, so the fields cannot run into one another: no concatenation of the
        # five values can be read as a different five. `ensure_ascii` keeps the payload pure
        # ASCII, which both escapes every code point injectively and stops a lone surrogate —
        # which a parser can produce and a `str` can hold — from raising on the encode below.
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class VectorState(StrEnum):
    """What a vector store found for one chunk, and therefore what has to happen next.

    A total answer: a store returns one of these for *every* chunk it was asked about, so a
    caller never has to decide what an absence meant. The three that are not ``READABLE`` are
    not interchangeable — they are the three groups the reuse partition reports separately,
    and collapsing them is how a report comes to claim that avoided work was avoided for a
    reason it was not.
    """

    READABLE = "readable"
    """A row whose recorded embedding input is this chunk's, with a vector that reads back."""

    STALE = "stale"
    """A row for this chunk id whose recorded embedding input is **not** this chunk's.

    The case chunk-id reuse gets wrong. The id survived, the embedded string did not.
    """

    CORRUPT = "corrupt"
    """A row claiming this chunk's embedding input whose vector cannot be used.

    Wrong dimension for the index, or a row that contradicts itself — a recorded identity that
    does not describe the chunk stored beside it. Identity metadata asserting that a usable
    vector exists is not the same as one existing, so this is checked rather than trusted.
    """

    ABSENT = "absent"
    """No row for this chunk id at all."""


class StoredVector(BaseModel):
    """A vector store's verdict on one chunk, and the vector when there is one to have.

    Returned by :meth:`~manicule.core.protocols.VectorStore.stored_vectors`, which is asked
    about *chunks* rather than ids precisely so that this verdict can be about the embedding
    input. A store handed only an id could answer nothing better than "a row exists".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: VectorState
    vector: tuple[float, ...] = ()
    """The stored vector, present only when :attr:`state` is
    :attr:`~VectorState.READABLE`. Already unit-length, as stored."""

    identity_recorded: bool = True
    """Whether the row carried a written identity, or one had to be reconstructed.

    ``False`` is a row that predates the identity column: its embedding input was recovered
    from the chunk stored beside the vector, and writing the row again records it. Counting
    these is how the one-time backfill is visible — the number an operator watches reach zero
    — without a scan of the table to produce it.
    """

    @property
    def is_reusable(self) -> bool:
        """Whether this vector can stand in for a forward pass."""
        return self.state is VectorState.READABLE


UNRECORDED_IDENTITY: Final = ""
"""What a stored row holds when it was written before identities were recorded.

Not a value :func:`embedding_input_identity` can ever produce — it returns a hex digest — so
it cannot be mistaken for a match. It means the identity has to be *reconstructed* from the
chunk stored beside the vector rather than read; see :func:`classify_stored_vector`.
"""


def classify_stored_vector(
    chunk: Chunk,
    *,
    recorded_identity: str,
    stored_embed_text: str | None,
    stored_vector: Sequence[float] | None,
    embed: EmbedFingerprint,
    middleware: Sequence[str] = (),
) -> StoredVector:
    """Decide whether one stored row is still this chunk's vector.

    The rule every :class:`~manicule.core.protocols.VectorStore` answers
    :meth:`~manicule.core.protocols.VectorStore.stored_vectors` with, written once so that two
    backends cannot come to two answers. A backend's job is to produce the three things a row
    knows — its recorded identity, the ``embed_text`` of the chunk stored beside it, and the
    vector — and this decides what they mean.

    The steps are ordered, and each rejects something the next would have accepted:

    1. **A row whose chunk cannot be read is corrupt.** Nothing about it can be checked, and a
       row that cannot be checked is not a vector to reuse.
    2. **A row whose recorded identity claims the input being asked for, while the chunk
       stored beside it says otherwise, is corrupt.** Metadata asserting that a usable vector
       exists is the one thing that must never be taken on trust, and here the row's two halves
       disagree about the very input the caller wants. A row whose recorded identity claims
       something *else* is merely stale, whatever its chunk says: a change of embedding
       fingerprint or of middleware declaration puts every row in that position at once, and
       calling a whole corpus corrupt would say the directory was damaged when the model
       simply changed.
    3. **A row about a different embedding input is stale.** The case chunk-id reuse gets
       wrong: the id survived a re-parse because ``text`` did, while the heading breadcrumb in
       ``embed_text`` did not.
    4. **A row of the wrong dimension or with a non-finite component is corrupt**, however
       current its identity says it is. ``NaN`` and infinities cannot be ranked meaningfully.
    5. What is left is readable.

    Args:
        chunk: The chunk the row is being offered for. A chunk rather than an id, because the
            question is about the embedding input and an id cannot answer it.
        recorded_identity: What the row stores, or :data:`UNRECORDED_IDENTITY`.
        stored_embed_text: The ``embed_text`` of the chunk stored beside the vector, or ``None``
            when the backend could not read one.
        stored_vector: The vector as stored, or ``None`` when there is not one.
        embed: The fingerprint the index was built with.
        middleware: The embed-text middleware declaration the store was prepared with.

    Returns:
        The verdict, carrying the vector only when it is
        :attr:`~VectorState.READABLE`.
    """
    if stored_embed_text is None:
        return StoredVector(state=VectorState.CORRUPT)

    wanted = embedding_input_identity(
        chunk.embed_text, document_id=chunk.document_id, embed=embed, middleware=middleware
    )
    derived = embedding_input_identity(
        stored_embed_text, document_id=chunk.document_id, embed=embed, middleware=middleware
    )
    recorded = recorded_identity or UNRECORDED_IDENTITY
    if recorded == UNRECORDED_IDENTITY:
        recorded = derived
    elif recorded != derived:
        claims_current = recorded == wanted
        return StoredVector(state=VectorState.CORRUPT if claims_current else VectorState.STALE)

    if recorded != wanted:
        return StoredVector(state=VectorState.STALE)
    if (
        stored_vector is None
        or len(stored_vector) != embed.dimension
        or not is_finite_vector(stored_vector)
    ):
        return StoredVector(state=VectorState.CORRUPT)
    return StoredVector(
        state=VectorState.READABLE,
        vector=tuple(float(value) for value in stored_vector),
        identity_recorded=bool(recorded_identity),
    )


def choose_stored_vector(
    by_id: StoredVector, by_identity: StoredVector | None = None
) -> StoredVector:
    """Pick between the row under a chunk's own id and any row holding the same input.

    **Why a second lookup exists at all.** A chunk id is derived from position as well as text
    (:func:`~manicule.core.ids.chunk_id`), so inserting one paragraph at the top of a document
    changes the id of every chunk below it while changing not one embedding input. Keyed on the
    id alone, that edit re-embeds the whole document to store vectors it already holds — the
    same waste this path exists to remove, arriving through the other door. A vector is a pure
    function of the embedding input under a fixed fingerprint, so a row recorded against that
    input *is* this chunk's vector, whichever id it happens to be filed under.

    **The row under the chunk's own id wins when it is usable**, because it is the row that
    would be overwritten and the one whose identity may need recording.

    **A corrupt row is never shopped around.** A present-and-damaged row is evidence about this
    chunk's stored vector, and the honest response to damage in a directory is to rebuild the
    row rather than to find another copy and leave the question of what damaged it unasked.
    Absent and stale carry no such evidence, so those are the two the fallback answers.

    Args:
        by_id: The verdict on the row stored under the chunk's id.
        by_identity: The verdict on a row found by embedding-input identity instead, when the
            store looked and found one.

    Returns:
        The verdict the caller should act on.
    """
    if by_id.state is not VectorState.ABSENT and by_id.state is not VectorState.STALE:
        return by_id
    if by_identity is not None and by_identity.is_reusable:
        return by_identity
    return by_id


def require_within_context(
    chunks: Sequence[Chunk],
    fingerprint: EmbedFingerprint,
    chunk_fingerprint: ChunkFingerprint | None = None,
) -> None:
    """Raise unless every chunk fits in what the embedder will actually read.

    **Every path that embeds chunks must call this**, and one path in particular: re-embed.
    Re-embedding reads stored ``embed_text`` and does not re-chunk, so the chunker's own
    budget refusal never runs. A model reconfigured to a shorter sequence length — a
    different checkpoint, an edited config, a changed backend default — leaves the
    embedding fingerprint identical, so no comparison fires, and every oversized chunk is
    quietly truncated and stored as a vector claiming text it never saw. This function is
    what stands between that and the index.

    It checks the property rather than a proxy for it, which is why
    :attr:`EmbedFingerprint.max_sequence_length` can stay out of identity: a limit that
    rises harms nothing, and a limit that falls is caught here on the batch it would have
    damaged.

    Args:
        chunks: What is about to be embedded. ``token_count`` is trusted, which is only
            sound if it was measured with the embedder's tokenizer — hence the next
            argument.
        fingerprint: The embedder about to receive them.
        chunk_fingerprint: The chunker that produced them, when it is known. Supplying it
            adds a tokenizer check, because a token count taken under a different
            vocabulary is not a measurement of anything relevant.

    Raises:
        ContextOverflowError: Any chunk exceeds the limit, or the token counts were taken
            with a different tokenizer. The message names the worst offenders rather than
            only the first, so one run fixes the batch.
    """
    mismatched_tokenizer = (
        chunk_fingerprint is not None
        and bool(fingerprint.tokenizer_id)
        and chunk_fingerprint.tokenizer_id != fingerprint.tokenizer_id
    )
    if mismatched_tokenizer and chunk_fingerprint is not None:
        msg = (
            f"chunks were counted with {chunk_fingerprint.tokenizer_id!r} but "
            f"{fingerprint.model_id} tokenizes with {fingerprint.tokenizer_id!r}. "
            f"Token counts taken under a different vocabulary cannot be checked against "
            f"this model's limit; re-chunk with the matching tokenizer."
        )
        raise ContextOverflowError(msg)

    limit = fingerprint.max_sequence_length
    oversized = sorted(
        (chunk for chunk in chunks if chunk.token_count > limit),
        key=lambda chunk: chunk.token_count,
        reverse=True,
    )
    if not oversized:
        return

    worst = oversized[:3]
    listed = ", ".join(f"{chunk.id} ({chunk.token_count} tokens)" for chunk in worst)
    more = f" and {len(oversized) - len(worst)} more" if len(oversized) > len(worst) else ""
    msg = (
        f"{len(oversized)} chunk(s) exceed the {limit}-token limit of "
        f"{fingerprint.describe()}: {listed}{more}. Embedding them would truncate the text "
        f"without an error and store vectors that describe less than the chunks claim. "
        f"Re-chunk against this model's limit, or embed with a model that fits the corpus."
    )
    raise ContextOverflowError(msg)


__all__ = [
    "EMBEDDING_IDENTITY_VERSION",
    "UNRECORDED_IDENTITY",
    "EmbedFingerprint",
    "IndexFingerprints",
    "NDArrayLike",
    "Pooling",
    "StoredVector",
    "TokenStates",
    "Vector",
    "VectorState",
    "choose_stored_vector",
    "classify_stored_vector",
    "embedding_input_identity",
    "is_finite_vector",
    "require_within_context",
]
