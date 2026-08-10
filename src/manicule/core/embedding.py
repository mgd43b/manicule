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

from manicule.core.fingerprints import Fingerprint

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

    Two fields are recorded and excluded from identity, for the same reason in both cases:
    they describe how the vectors were made without changing whether they are comparable.
    :attr:`backend` is the runtime, and the same model under two runtimes produces
    interchangeable vectors. :attr:`max_sequence_length` constrains what text can be
    embedded, which is the chunker's problem and belongs to
    :class:`~manicule.core.fingerprints.ChunkFingerprint`.
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
        "the value the chunk budget is checked against: beyond it the input is truncated "
        "with no error raised, so an over-budget chunk is indexed as its opening tokens "
        "while still claiming all of its text — a citation quoting words the index never saw. "
        "Note this is the *effective* limit, which is not always the positional-embedding "
        "limit: some widely used models ship configured well below what their architecture "
        "allows, so it must be read from the loaded model rather than assumed.",
    )

    backend: str = Field(
        default="",
        description="Which runtime produced the vectors, e.g. ``mlx`` or ``onnx``.",
    )

    @override
    def describe(self) -> str:
        """A one-line human-readable form, for error messages and diagnostics."""
        revision = f"@{self.revision}" if self.revision else ""
        norm = "normalized" if self.normalized else "unnormalized"
        return f"{self.model_id}{revision} ({self.dimension}d, {self.pooling.value}, {norm})"


__all__ = [
    "EmbedFingerprint",
    "NDArrayLike",
    "Pooling",
    "TokenStates",
    "Vector",
]
