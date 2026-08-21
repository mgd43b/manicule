"""Embedding vocabulary: vectors, token states, and the fingerprint that identifies them.

**Nothing in manicule may hardcode a vector dimension.** The dimension is read from
:class:`EmbedFingerprint` at run time, the vector table is created at first ingest with
whatever the embedder reports, and ingest refuses to start when the fingerprint does not
match the one the index was built with.

Dimension alone is not identity. Two unrelated 1024-dimension models produce vectors that
are mutually meaningless, and a guard that compared only ``D`` would wave that through
while quietly destroying retrieval quality. :meth:`EmbedFingerprint.identity` is what gets
compared.

Identity is not integrity either. Every provenance check here compares a row against
*metadata*, and a bit flip that turns one finite component into another finite component
leaves all of it intact. :func:`vector_checksum` is the third axis: a versioned digest over
the exact numbers a backend persists, recomputed on readback. What it establishes, and the
two things it deliberately does not, are on :func:`verify_stored_checksum`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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

FLOAT32_EPSILON: Final = 2.0**-23
"""The spacing of float32 values either side of one."""


def canonical_stored_vector(vector: Vector) -> tuple[float, ...]:
    """Return the exact normalized float32 values a vector backend persists.

    Publication identities are computed before Lance writes and reused vectors are read after
    it writes. Canonicalizing at both boundaries prevents normalization and float32 rounding
    from turning an identical retry into a different publication.
    """
    values = [float(value) for value in vector]
    try:
        float32_inputs = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in values)
    except OverflowError as exc:
        msg = (
            "embedding vector became non-finite when converted to float32; NaN and infinity "
            "cannot participate in cosine distance"
        )
        raise ValueError(msg) from exc
    if not is_finite_vector(float32_inputs):
        msg = (
            "embedding vector became non-finite when converted to float32; NaN and infinity "
            "cannot participate in cosine distance"
        )
        raise ValueError(msg)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm != 0.0 and abs(norm - 1.0) >= FLOAT32_EPSILON:
        values = [value / norm for value in values]
    canonical = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in values)
    if not is_finite_vector(canonical):
        msg = (
            "embedding vector became non-finite when canonicalized to float32; NaN and infinity "
            "cannot participate in cosine distance"
        )
        raise ValueError(msg)
    return canonical


VECTOR_CHECKSUM_VERSION: Final = "1"
"""The checksum format this build writes. See :func:`vector_checksum` for what it names.

Stored beside every checksum rather than assumed, so that changing the rule is a migration
somebody can execute instead of a silent reinterpretation of every digest already written. A
row recording a version this build does not know is refused as
:attr:`VectorIntegrity.UNKNOWN_VERSION` — not recomputed under the current rule, which would
be this build deciding that an older build's digest meant what this one means.
"""

VECTOR_CHECKSUM_DOMAINS: Final[dict[str, bytes]] = {
    "1": b"manicule/vector-checksum/1\x1f",
}
"""The domain separator each format version prefixes its preimage with.

A digest is only meaningful inside the domain it was taken for. The prefix is what stops a
SHA-256 of the same bytes taken somewhere else in this repository — an evidence digest, a
blob digest, a future keyed variant — from being accepted here, and terminating it with a
byte that cannot begin a length field is what stops one version's separator being a prefix of
another's.
"""

UNRECORDED_CHECKSUM: Final = ""
"""What a row holds when it was written before checksums were recorded.

Not a value :func:`vector_checksum` can produce — it returns 64 hex characters — so it cannot
be mistaken for a match. It means the row predates the contract and its numerical integrity is
*unverified* rather than *bad*; see :class:`VectorIntegrity` for the difference and
``docs/storage.md`` §6.2.5 for the compatibility policy that rests on it.
"""

_CHECKSUM_DIGITS: Final = 64
"""Hex characters in a SHA-256 digest. A recorded value of any other length is malformed."""


class VectorIntegrity(StrEnum):
    """What was established about one stored vector's *numbers*, as opposed to its provenance.

    A second axis beside :class:`VectorState`, and deliberately not folded into it. ``STATE``
    answers "can this vector stand in for a forward pass"; this answers "what did the
    numerical check actually find", which is the question an operator has to answer to know
    whether they are looking at an upgrade in progress or a damaged directory. Collapsing the
    two would make a table that has never been backfilled indistinguishable from one that has
    been corrupted, and those call for opposite responses.

    Every member except :attr:`VERIFIED` and :attr:`UNVERIFIED` is a refusal.
    """

    NOT_CHECKED = "not_checked"
    """No numerical check was reached — there is no row, or the row was refused before it."""

    VERIFIED = "verified"
    """A recorded checksum of a known version was recomputed from the stored vector and matched."""

    UNVERIFIED = "unverified"
    """The row records no checksum, and none was required of it.

    A row written before the contract existed. Readable, counted, and cleared by the backfill
    — never silently reported as verified.
    """

    MISSING = "missing"
    """No checksum was recorded where one was required.

    The refusal a checksum-required publication makes. The same physical row that reads as
    :attr:`UNVERIFIED` for ordinary reuse reads as this when a new generation is being
    validated, because "we have not checked it yet" is not a thing to publish.
    """

    MALFORMED = "malformed"
    """A recorded checksum or version that is not well formed, so nothing can be compared."""

    UNKNOWN_VERSION = "unknown_version"
    """A checksum recorded under a format version this build does not implement."""

    MISMATCHED = "mismatched"
    """The recomputed checksum is not the recorded one. The corruption this contract exists for."""

    WRONG_DIMENSION = "wrong_dimension"
    """The stored vector is not the dimension the index was built for."""

    NON_FINITE = "non_finite"
    """The stored vector holds ``NaN`` or an infinity, which cannot be ranked."""

    UNREADABLE = "unreadable"
    """The row carries no vector this store could read at all."""

    @property
    def accepts(self) -> bool:
        """Whether a row in this condition may still be read, reused or published."""
        return self in {VectorIntegrity.VERIFIED, VectorIntegrity.UNVERIFIED}


def vector_checksum(stored: Vector, *, version: str = VECTOR_CHECKSUM_VERSION) -> str:
    """The versioned checksum of the exact numerical representation ``stored`` persists as.

    **It hashes what storage holds, never what a model returned.** Pass either the output of
    :func:`canonical_stored_vector` — which is what a write stores — or the values a backend
    read back. Handing it raw model output would hash a representation that is not the one on
    disk, and the digest would then fail against the row it was supposed to describe. That is
    the whole trap this signature is shaped to avoid: there is no "normalize it for me" mode,
    because the canonical conversion belongs to exactly one function and it is above.

    The contract, versioned as :data:`VECTOR_CHECKSUM_VERSION` and recorded beside every value:

    ``domain``
        :data:`VECTOR_CHECKSUM_DOMAINS` for the format version, first, so a digest of the same
        bytes taken for another purpose in this repository cannot be accepted here.

    ``dimension``
        Eight bytes, big-endian, unsigned, before the components. Fixed-width and up front, so
        a ``d``-component preimage can never equal a differently-shaped one — without it the
        empty vector and a vector of nothing but a length prefix would collide.

    ``components``
        IEEE-754 ``binary32``, big-endian, in index order, one after another. ``binary32``
        because that is the persisted dtype: the Lance column is
        ``fixed_size_list<float32, d>``, so hashing ``float64`` would hash a representation
        that does not exist on disk. Big-endian because the preimage is a *defined*
        serialization rather than a memory image — the same vector then checksums identically
        on a little-endian and a big-endian host, which is what makes the digest portable
        rather than a property of the machine that wrote it.

    ``signed zero``
        ``-0.0`` is hashed as ``+0.0``. The two are numerically equal, rank identically under
        every metric, and differ only in a sign bit that a storage layer, an Arrow round trip
        or a copy is free to drop. Hashing them apart would make a *representation* change
        report as a *value* change, which is the one thing an integrity check must not do.

    ``non-finite values``
        Refused rather than hashed. ``NaN`` and infinities are rejected before storage and
        rejected again here, so a checksum can never certify a vector that cannot be ranked.

    **No embedding fingerprint or vector identity is in the preimage, and that is deliberate.**
    Provenance is already checked, separately and mandatorily, by
    :func:`embedding_input_identity` and by the fingerprint comparison the store performs; a
    digest that mixed the two would report one failure for two unrelated causes and would make
    the same bytes checksum differently in two publications, so a copy could not be verified
    without rehashing it under a new scope. Numerical integrity is about the numbers. What this
    buys and what it does not is stated on :func:`verify_stored_checksum`.

    Args:
        stored: The values as they are, or are about to be, persisted.
        version: Which format to compute. Present so a future version can be computed
            deliberately; the default is the one new rows record.

    Returns:
        64 lowercase hex characters.

    Raises:
        ValueError: ``version`` is not implemented, or a component is not finite.
    """
    domain = VECTOR_CHECKSUM_DOMAINS.get(version)
    if domain is None:
        msg = (
            f"vector checksum format {version!r} is not implemented by this build; "
            f"known formats are {sorted(VECTOR_CHECKSUM_DOMAINS)}"
        )
        raise ValueError(msg)
    values = [float(value) for value in stored]
    if not is_finite_vector(values):
        msg = (
            "a non-finite component cannot be checksummed: NaN and infinity are refused "
            "before storage, so certifying one here would vouch for a vector that cannot "
            "participate in cosine distance"
        )
        raise ValueError(msg)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(values).to_bytes(8, "big"))
    # `value == 0.0` is true of `-0.0`, which is the whole point: the sign bit of a zero is a
    # representation detail and is normalized away rather than hashed.
    digest.update(
        struct.pack(f">{len(values)}f", *(0.0 if value == 0.0 else value for value in values))
    )
    return digest.hexdigest()


def verify_stored_checksum(
    stored: Vector, *, recorded: str, version: str, required: bool
) -> VectorIntegrity:
    """Decide what one row's recorded checksum establishes about its stored vector.

    **What a match proves.** The bytes on disk are the bytes that were written. A bit flip, a
    truncated write, a storage-layer rewrite or a copy that dropped a component changes the
    recomputed digest and is refused here, at the cost of one SHA-256 over ``4d`` bytes rather
    than a forward pass through the model.

    **What a match does not prove**, and neither of these is a gap this can close:

    - *It is not semantic correctness.* Nothing here says the model produced the right vector
      for the text. The only check that could is re-embedding, which is the cost this exists
      to avoid.
    - *It is not tamper resistance.* The digest is unkeyed and stored beside what it describes,
      so anything able to rewrite the vector can rewrite the checksum. Defending against that
      needs a keyed or externally anchored design, and ``docs/storage.md`` §6.2.5 records it as
      out of scope rather than quietly implied.

    Args:
        stored: The vector as the backend read it.
        recorded: The checksum column, or :data:`UNRECORDED_CHECKSUM`.
        version: The checksum-format column, or :data:`UNRECORDED_CHECKSUM`.
        required: Whether a row without a checksum is a refusal. ``True`` at a publication
            fence, where an unverified row must not be published as a verified one; ``False``
            on the ordinary read path, where a row predating the contract stays readable.

    Returns:
        The verdict, which is :attr:`VectorIntegrity.VERIFIED` or
        :attr:`~VectorIntegrity.UNVERIFIED` when the row may be used and names the refusal
        otherwise.
    """
    if not recorded and not version:
        return VectorIntegrity.MISSING if required else VectorIntegrity.UNVERIFIED
    if not recorded or not version:
        # Half a record is not a lesser record: one column without the other is a row whose two
        # halves were not written together, and nothing can be compared against it.
        return VectorIntegrity.MALFORMED
    if version not in VECTOR_CHECKSUM_DOMAINS:
        return VectorIntegrity.UNKNOWN_VERSION
    if len(recorded) != _CHECKSUM_DIGITS or any(
        character not in "0123456789abcdef" for character in recorded
    ):
        return VectorIntegrity.MALFORMED
    try:
        computed = vector_checksum(stored, version=version)
    except ValueError:  # pragma: no cover - callers reject non-finite vectors first
        return VectorIntegrity.NON_FINITE
    # Constant time is not what this is for — the digest is unkeyed and public to anything that
    # can read the row — but comparing digests with `compare_digest` costs nothing and keeps one
    # habit rather than two.
    return (
        VectorIntegrity.VERIFIED
        if hmac.compare_digest(computed, recorded)
        else VectorIntegrity.MISMATCHED
    )


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
        description="Which vector table or named generation holds the vectors. A pointer "
        "rather than a constant, because a re-embed builds its replacement alongside the "
        "live one and atomically moves the pointer after validation.",
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

    integrity: VectorIntegrity = VectorIntegrity.NOT_CHECKED
    """What the numerical check found, on the axis :attr:`state` does not describe.

    :attr:`~VectorIntegrity.VERIFIED` and :attr:`~VectorIntegrity.UNVERIFIED` are the two a
    :attr:`~VectorState.READABLE` verdict can carry, and the difference between them is a
    corpus mid-upgrade versus one whose checksums are all present. Everything else is why the
    verdict is :attr:`~VectorState.CORRUPT`, kept apart so a diagnostic can say *which*
    numerical check failed rather than only that one did.
    """

    @property
    def is_reusable(self) -> bool:
        """Whether this vector can stand in for a forward pass."""
        return self.state is VectorState.READABLE

    @property
    def checksum_recorded(self) -> bool:
        """Whether this row's numbers were checked against a checksum it carried.

        ``False`` for a readable row means the checksum backfill has not reached it yet —
        the number an operator watches reach zero, on the same terms as
        :attr:`identity_recorded` and for the same reason.
        """
        return self.integrity is VectorIntegrity.VERIFIED


@dataclass(frozen=True, slots=True)
class VectorChecksumCoverage:
    """How much of one vector table's numerical integrity is established, and how firmly.

    Assembled by the storage layer and reported verbatim by ``status``, ``doctor`` and the
    checksum command. Counts only: no checksum value, no vector component, no chunk text and no
    document identifier leaves here, because the surfaces that render it are reachable by an
    assistant and by whatever a shell pipeline points at.

    **:attr:`recomputed` is what keeps this honest.** Counting how many rows *carry* a checksum
    costs two predicates and is what a status page can afford on every call; recomputing every
    digest costs a scan of the corpus and is what the operator asks for deliberately. Both are
    called "coverage" in ordinary speech and they establish very different things, so the report
    says which one it is rather than letting a cheap count read as a clean bill of health.
    """

    rows: int = 0
    recorded: int = 0
    """Rows carrying a checksum, whether or not it was recomputed."""

    verified: int = 0
    """Rows whose checksum was recomputed from the stored vector and matched.

    Zero when :attr:`recomputed` is false — nothing was verified, and saying so is the point.
    """

    failed: int = 0
    """Rows a recomputed checksum refused. See :attr:`failures` for the split by kind."""

    failures: Mapping[str, int] = field(default_factory=dict[str, int])
    """``VectorIntegrity`` value to count, for the refusals only.

    A bounded typed diagnostic: at most one entry per enum member, each an integer, so the
    worst this can say about a corpus is which kinds of damage it holds and how much of each.
    """

    recomputed: bool = False
    """Whether the digests were recomputed, or the rows carrying one were merely counted."""

    scanned: bool = True
    """Whether the numbers describe a table at all.

    ``False`` where there was nothing to look at — no vector table, or a backend without the
    capability — so a surface can distinguish "nothing is wrong" from "nothing was looked at".
    A report that answered ``0`` for both would be the more dangerous of the two lies.
    """

    @property
    def unverified(self) -> int:
        """Rows recording no checksum. Readable; cleared by the backfill, never by a re-embed."""
        return max(0, self.rows - self.recorded)

    @property
    def complete(self) -> bool:
        """Whether every row carries a checksum and none of the recorded ones was refused."""
        return self.scanned and self.unverified == 0 and self.failed == 0

    @property
    def fraction(self) -> float:
        """Rows carrying a checksum over all rows, in ``[0, 1]``; ``1.0`` over an empty table."""
        return 1.0 if self.rows == 0 else self.recorded / self.rows


@dataclass(frozen=True, slots=True)
class VectorChecksumBackfill:
    """What one bounded pass of the checksum backfill did.

    Resumable and idempotent by construction rather than by bookkeeping: the pass selects rows
    that record no checksum, so a row it finished is not a row the next pass can see. There is
    no cursor to lose, and an interruption costs at most the page in flight.
    """

    scanned: int = 0
    """Rows without a checksum that this pass read."""

    written: int = 0
    """Rows this pass gave a checksum. Never more than :attr:`scanned`."""

    unhashable: int = 0
    """Rows whose stored vector could not be checksummed at all — non-finite, or unreadable.

    Left exactly as they were. A checksum computed over a vector that is already damaged would
    certify the damage, which is worse than leaving the row honestly unverified.
    """

    remaining: int = 0
    """Rows still recording no checksum after this pass. Zero is a finished backfill."""

    dry_run: bool = False
    """Whether this is a report of what a pass would do rather than of what one did."""

    @property
    def done(self) -> bool:
        """Whether another pass would find nothing to do."""
        return self.remaining == 0


UNRECORDED_IDENTITY: Final = ""
"""What a stored row holds when it was written before identities were recorded.

Not a value :func:`embedding_input_identity` can ever produce — it returns a hex digest — so
it cannot be mistaken for a match. It means the identity has to be *reconstructed* from the
chunk stored beside the vector rather than read; see :func:`classify_stored_vector`.
"""


def classify_stored_vector(  # noqa: PLR0911 - one return per rejection, in the documented order
    chunk: Chunk,
    *,
    recorded_identity: str,
    stored_embed_text: str | None,
    stored_vector: Sequence[float] | None,
    embed: EmbedFingerprint,
    middleware: Sequence[str] = (),
    recorded_checksum: str = UNRECORDED_CHECKSUM,
    recorded_checksum_version: str = UNRECORDED_CHECKSUM,
    require_checksum: bool = False,
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
    5. **A row whose recorded checksum does not describe the vector stored beside it is
       corrupt**, and this is the step the four above cannot make. Every one of them compares a
       row against metadata; a bit flip that turns one finite component into another finite
       component leaves the chunk id, the embedding input, the fingerprint, the dimension and
       the publication all intact, so nothing before here has anything to disagree with. The
       checksum is recomputed from the stored numbers and compared to what the row recorded —
       ``docs/storage.md`` §6.2.5 for what that does and does not establish. A row predating
       the contract records no checksum and stays readable unless ``require_checksum``, which
       is what a publication fence passes.
    6. What is left is readable.

    Args:
        chunk: The chunk the row is being offered for. A chunk rather than an id, because the
            question is about the embedding input and an id cannot answer it.
        recorded_identity: What the row stores, or :data:`UNRECORDED_IDENTITY`.
        stored_embed_text: The ``embed_text`` of the chunk stored beside the vector, or ``None``
            when the backend could not read one.
        stored_vector: The vector as stored, or ``None`` when there is not one.
        embed: The fingerprint the index was built with.
        middleware: The embed-text middleware declaration the store was prepared with.
        recorded_checksum: The row's checksum column, or :data:`UNRECORDED_CHECKSUM`.
        recorded_checksum_version: The row's checksum-format column, or
            :data:`UNRECORDED_CHECKSUM`.
        require_checksum: Whether a row carrying no checksum is a refusal. ``False`` on the
            ordinary read path, so an upgrade does not declare an existing corpus corrupt;
            ``True`` at a publication fence, so an unverified row is never published as a
            verified one.

    Returns:
        The verdict, carrying the vector only when it is
        :attr:`~VectorState.READABLE`, and always carrying the numerical finding in
        :attr:`~StoredVector.integrity`.
    """
    if stored_embed_text is None:
        return StoredVector(state=VectorState.CORRUPT, integrity=VectorIntegrity.UNREADABLE)

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
    if stored_vector is None:
        return StoredVector(state=VectorState.CORRUPT, integrity=VectorIntegrity.UNREADABLE)
    if len(stored_vector) != embed.dimension:
        return StoredVector(state=VectorState.CORRUPT, integrity=VectorIntegrity.WRONG_DIMENSION)
    if not is_finite_vector(stored_vector):
        return StoredVector(state=VectorState.CORRUPT, integrity=VectorIntegrity.NON_FINITE)
    integrity = verify_stored_checksum(
        stored_vector,
        recorded=recorded_checksum,
        version=recorded_checksum_version,
        required=require_checksum,
    )
    if not integrity.accepts:
        return StoredVector(state=VectorState.CORRUPT, integrity=integrity)
    return StoredVector(
        state=VectorState.READABLE,
        vector=tuple(float(value) for value in stored_vector),
        identity_recorded=bool(recorded_identity),
        integrity=integrity,
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
        chunks: What is about to be embedded. This boundary evaluates the supplied
            ``token_count``; callers whose embedder exposes exact counting remeasure
            ``embed_text`` immediately before calling it, while also preserving the
            conservative stored-count model-context check.
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

    if chunk_fingerprint is not None:
        budget = chunk_fingerprint.max_tokens
        outside_budget = [chunk.token_count for chunk in chunks if chunk.token_count > budget]
        if outside_budget:
            msg = (
                f"{len(outside_budget)} chunk(s) exceed their fingerprinted {budget}-token "
                f"final embed_text budget; the maximum is {max(outside_budget)} tokens. "
                f"A larger model context does not change the chunk policy. Rechunk from "
                f"retained source bytes into a replacement generation."
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
    "UNRECORDED_CHECKSUM",
    "UNRECORDED_IDENTITY",
    "VECTOR_CHECKSUM_DOMAINS",
    "VECTOR_CHECKSUM_VERSION",
    "EmbedFingerprint",
    "IndexFingerprints",
    "NDArrayLike",
    "Pooling",
    "StoredVector",
    "TokenStates",
    "Vector",
    "VectorChecksumBackfill",
    "VectorChecksumCoverage",
    "VectorIntegrity",
    "VectorState",
    "canonical_stored_vector",
    "choose_stored_vector",
    "classify_stored_vector",
    "embedding_input_identity",
    "is_finite_vector",
    "require_within_context",
    "vector_checksum",
    "verify_stored_checksum",
]
