"""Fingerprints — the identity of a process that produced stored data.

Two exist: :class:`~manicule.core.embedding.EmbedFingerprint` for vectors, and
:class:`ChunkFingerprint` for chunk boundaries. Both are persisted alongside the index and
compared before anything is written, because both describe transformations whose output is
useless when mixed with output from a different version of themselves — and useless in the
quiet way, where nothing raises and every answer is slightly wrong.

They share these semantics deliberately:

Comparison is on a canonical serialisation, byte for byte
    Not on one field. A guard that compared only a dimension, or only a version number,
    passes exactly the cases that matter — two different models at the same dimension, two
    different grammars at the same chunk size.

Identity is a declared subset of the fields
    Some fields describe the producer without affecting its output. Those are recorded for
    diagnostics and excluded from comparison, so that moving between machines does not
    invalidate a corpus for no reason.

A mismatch raises, always
    Never a warning. There is nothing downstream that can detect mixed output, so the only
    place it can be caught is here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar, Self, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.core.errors import FingerprintMismatchError


class Fingerprint(BaseModel):
    """Base for fingerprints. Subclasses declare which of their fields are identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ()

    def identity(self) -> dict[str, JsonValue]:
        """The fields that decide comparability, as JSON-shaped values."""
        dumped: dict[str, JsonValue] = self.model_dump(mode="json")
        return {name: dumped[name] for name in self.IDENTITY_FIELDS}

    def canonical(self) -> str:
        """A stable serialisation of :meth:`identity`.

        Keys sorted and separators fixed, so two runs of the same version produce byte-equal
        output and a stored fingerprint can be compared without being parsed.
        """
        return json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))

    def matches(self, other: Self) -> bool:
        """Whether ``other`` produces output interchangeable with this one's."""
        return type(self) is type(other) and self.canonical() == other.canonical()

    def changed_fields(self, other: Self) -> frozenset[str]:
        """Which identity fields differ.

        What makes selective invalidation possible: a grammar upgrade that changes one
        language should invalidate the documents in that language, not the corpus.
        """
        mine = self.identity()
        theirs = other.identity()
        return frozenset(name for name in (*mine, *theirs) if mine.get(name) != theirs.get(name))

    def require_match(self, other: Self) -> None:
        """Raise unless ``other`` is interchangeable with this one.

        Args:
            other: The fingerprint offered by whatever is about to read or write.

        Raises:
            FingerprintMismatchError: When they differ, naming the fields that differ.
        """
        if self.matches(other):
            return
        changed = ", ".join(sorted(self.changed_fields(other))) or "type"
        msg = (
            f"{type(self).__name__} mismatch on {changed}: the index was built with "
            f"{self.describe()}, but {other.describe()} was offered. Re-run against the "
            f"index this produced, or rebuild the index."
        )
        raise FingerprintMismatchError(msg)

    def describe(self) -> str:
        """A one-line human-readable form, for error messages and diagnostics."""
        return self.canonical()


class ChunkFingerprint(Fingerprint):
    """The identity of the process that decided where chunks begin and end.

    Persisted per corpus and recorded per document, so a change can be traced to the
    documents it affects. Re-chunking is cheaper than re-embedding but not free, and
    invalidating everything because one grammar moved is the expensive mistake.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "chunker",
        "version",
        "max_tokens",
        "overlap_tokens",
        "tokenizer_id",
        "grammars",
        "embed_text_middleware",
    )

    chunker: str = Field(min_length=1, description="Registered chunker that produced the chunks.")
    version: str = Field(min_length=1, description="The chunker's own version.")

    max_tokens: int = Field(
        gt=0,
        description="Token budget per chunk, measured on ``embed_text``. Must not exceed the "
        "embedder's ``max_sequence_length``: past that limit the text is truncated with no "
        "error, and the chunk is indexed as its first N tokens while still claiming all of "
        "its text — a citation that quotes words the index never saw. The chunker enforces "
        "this when it runs; re-embed does not re-chunk, so that path enforces it with "
        "``require_within_context`` instead.",
    )
    overlap_tokens: int = Field(
        default=0, ge=0, description="Tokens repeated between adjacent chunks."
    )
    tokenizer_id: str = Field(
        min_length=1,
        description="Which tokenizer counted the tokens. A budget is meaningless without it: "
        "the same text is a different number of tokens under a different vocabulary.",
    )
    grammars: dict[str, str] = Field(
        default_factory=dict,
        description="Version by language for structure-aware chunking, e.g. "
        "``{'python': '0.21.0'}``. Recorded per language so that upgrading one grammar "
        "invalidates the documents in that language and leaves the rest alone.",
    )
    embed_text_middleware: tuple[str, ...] = Field(
        default=(),
        description="Sorted ``name@version`` for every middleware declaring "
        "``mutates_embedded_text``. Empty on a chunker's own fingerprint; the ingest "
        "pipeline folds the configured set in before comparing. Without it, two instances "
        "with identical configuration and different middleware produce different vectors "
        "from identical source bytes and **neither fingerprint refusal notices**, because "
        "neither otherwise knows middleware exists. Adding, removing or upgrading one is "
        "then exactly as loud as changing the chunk budget, which is what it is.",
    )

    def with_middleware(self, declarations: Sequence[str]) -> ChunkFingerprint:
        """This fingerprint, with ``declarations`` folded into its identity.

        Sorted and de-duplicated here rather than at every call site, so the identity does
        not depend on the order configuration happened to list middleware in — which is a
        legitimate thing to change without changing a single vector.
        """
        return self.model_copy(update={"embed_text_middleware": tuple(sorted(set(declarations)))})

    @override
    def describe(self) -> str:
        mutating = (
            f", mutated by {', '.join(self.embed_text_middleware)}"
            if self.embed_text_middleware
            else ""
        )
        return (
            f"{self.chunker} {self.version} ({self.max_tokens}+{self.overlap_tokens} tokens, "
            f"{self.tokenizer_id}{mutating})"
        )


__all__ = ["ChunkFingerprint", "Fingerprint"]
