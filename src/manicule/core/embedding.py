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

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import ClassVar, Protocol, override, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.content import Chunk
from manicule.core.errors import ContextOverflowError
from manicule.core.fingerprints import ChunkFingerprint, Fingerprint

type Vector = Sequence[float]
"""A finished embedding.

Deliberately a plain sequence: vectors cross a storage boundary, where a concrete,
serialisable value is what is wanted. Backends holding native arrays convert at the seam.
"""


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

    **:attr:`backend` is excluded because the runtimes were measured to agree.** The same
    model under MLX and under ONNX produces interchangeable vectors, which keeps a corpus
    portable between machines instead of demanding a re-embed on arrival. That is now a
    measurement rather than an assumption: ``tests/test_embedding_backends.py`` embeds the
    same texts through both runtimes and fails if any pair falls outside the tolerance stated
    there. **If parity ever stops holding, this exclusion is the decision to revisit** —
    moving ``backend`` into ``IDENTITY_FIELDS`` makes a runtime change a loud error with a
    re-embed path, which is the correct behaviour if the vectors really do differ.

    **:attr:`weights_ref` is excluded for the same reason and needs the same care.** A
    backend rarely runs the canonical repository's own files: MLX needs safetensors and
    ``BAAI/bge-m3`` publishes only a PyTorch pickle, so the weights actually executed come
    from a conversion. Recording which conversion ran is the difference between a diagnosable
    index and a mystery. It stays out of identity because a faithful re-encoding of the same
    weights is the thing parity certifies — and because the one re-encoding that *does* move
    the vectors, quantisation, is refused at load time rather than absorbed here: 4-bit
    ``bge-m3`` sits at cosine 0.92-0.97 to the same model in fp16, which is a different vector
    space wearing the same name.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_id",
        "revision",
        "dimension",
        "pooling",
        "normalized",
        "tokenizer_id",
    )

    model_id: str = Field(min_length=1, description="Model identifier, e.g. ``BAAI/bge-m3``.")
    revision: str | None = Field(
        default=None,
        description="Model revision or commit. ``None`` means unpinned, which is recorded "
        "faithfully rather than filled in with a guess.",
    )
    dimension: int = Field(gt=0, description="Read from the model. Never a literal.")
    pooling: Pooling
    normalized: bool = Field(description="Whether vectors are L2-normalised on output.")

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
        description="The artefact whose bytes the backend actually executed, when that is "
        "not the model's own repository — for example ``mlx-community/bge-m3-mlx-fp16`` for "
        "``BAAI/bge-m3`` under MLX, which publishes no safetensors. Recorded so that a "
        "vector can be traced to the weights that made it. Excluded from identity — see the "
        "class docstring.",
    )

    @override
    def describe(self) -> str:
        """A one-line human-readable form, for error messages and diagnostics."""
        revision = f"@{self.revision}" if self.revision else ""
        norm = "normalized" if self.normalized else "unnormalized"
        return f"{self.model_id}{revision} ({self.dimension}d, {self.pooling.value}, {norm})"


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
    "EmbedFingerprint",
    "NDArrayLike",
    "Pooling",
    "TokenStates",
    "Vector",
    "require_within_context",
]
