"""Fingerprints, and the guard that stops two models sharing one index.

The failure these prevent is silent by nature: mixed vectors write cleanly, search
successfully, and return answers drawn from a space the query does not live in.
"""

from __future__ import annotations

from typing import cast

import pytest

from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.errors import FingerprintMismatchError
from manicule.core.fingerprints import ChunkFingerprint


def embed(**overrides: object) -> EmbedFingerprint:
    base: dict[str, object] = {
        "model_id": "BAAI/bge-m3",
        "dimension": 1024,
        "pooling": Pooling.MEAN,
        "normalized": True,
        "tokenizer_id": "BAAI/bge-m3",
        "max_sequence_length": 512,
    }
    return EmbedFingerprint.model_validate({**base, **overrides})


def chunks(**overrides: object) -> ChunkFingerprint:
    base: dict[str, object] = {
        "chunker": "structural",
        "version": "1.0.0",
        "max_tokens": 512,
        "overlap_tokens": 64,
        "tokenizer_id": "BAAI/bge-m3",
    }
    return ChunkFingerprint.model_validate({**base, **overrides})


def test_two_different_models_of_the_same_size_do_not_match() -> None:
    """The case a dimension check waves through, and the reason identity is not one field."""
    built_with = embed(model_id="model-a")
    offered = embed(model_id="model-b")

    assert built_with.dimension == offered.dimension
    assert not built_with.matches(offered)
    with pytest.raises(FingerprintMismatchError, match="model_id"):
        built_with.require_match(offered)


@pytest.mark.parametrize(
    "difference",
    [
        {"dimension": 768},
        {"pooling": Pooling.CLS},
        {"normalized": False},
        {"revision": "abc123"},
        {"tokenizer_id": "other/tokenizer"},
    ],
    ids=lambda d: next(iter(d)),
)
def test_every_identity_field_invalidates_an_index(difference: dict[str, object]) -> None:
    with pytest.raises(FingerprintMismatchError):
        embed().require_match(embed(**difference))


def test_pooling_alone_is_enough_to_invalidate() -> None:
    """CLS and mean of the same states differ by around 0.86 cosine, and nothing raises."""
    assert not embed(pooling=Pooling.CLS).matches(embed(pooling=Pooling.MEAN))


def test_the_runtime_does_not_invalidate_an_index() -> None:
    """The same model under two runtimes produces interchangeable vectors.

    Including the runtime would force a full re-embed on moving between machines, for no
    gain in correctness.
    """
    assert embed(backend="mlx").matches(embed(backend="onnx"))


def test_the_sequence_limit_does_not_invalidate_an_index() -> None:
    """It constrains what can be embedded, not whether the results are comparable."""
    assert embed(max_sequence_length=512).matches(embed(max_sequence_length=8192))


def test_the_sequence_limit_is_required() -> None:
    """Unknown is the state that causes silent truncation, so it is not expressible."""
    with pytest.raises(ValueError, match="max_sequence_length"):
        EmbedFingerprint.model_validate(
            {
                "model_id": "m",
                "dimension": 8,
                "pooling": Pooling.MEAN,
                "normalized": True,
            }
        )


def test_the_canonical_form_is_byte_stable() -> None:
    """Storage compares the serialised form, so field order must not matter."""
    one = EmbedFingerprint.model_validate(
        {
            "model_id": "m",
            "dimension": 8,
            "pooling": "mean",
            "normalized": True,
            "max_sequence_length": 512,
        }
    )
    other = EmbedFingerprint.model_validate(
        {
            "max_sequence_length": 512,
            "normalized": True,
            "pooling": "mean",
            "dimension": 8,
            "model_id": "m",
        }
    )
    assert one.canonical() == other.canonical()
    assert "max_sequence_length" not in one.canonical()


def test_a_grammar_upgrade_names_only_what_changed() -> None:
    """Selective invalidation: one language moved, so one language's documents are stale."""
    before = chunks(grammars={"python": "0.21.0", "rust": "0.21.0"})
    after = chunks(grammars={"python": "0.22.0", "rust": "0.21.0"})

    assert before.changed_fields(after) == {"grammars"}
    assert before.grammars["rust"] == after.grammars["rust"]
    with pytest.raises(FingerprintMismatchError, match="grammars"):
        before.require_match(after)


def test_chunk_budgets_and_tokenizers_are_identity() -> None:
    assert not chunks(max_tokens=256).matches(chunks(max_tokens=512))
    assert not chunks(overlap_tokens=0).matches(chunks(overlap_tokens=64))
    assert not chunks(tokenizer_id="a").matches(chunks(tokenizer_id="b"))


def test_a_fingerprint_never_matches_one_of_another_kind() -> None:
    """They are stored side by side; comparing the wrong pair must not read as agreement."""
    embedding = embed()
    chunking = cast("EmbedFingerprint", chunks())
    assert not embedding.matches(chunking)
    with pytest.raises(FingerprintMismatchError):
        embedding.require_match(chunking)
