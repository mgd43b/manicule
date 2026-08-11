"""Fingerprints, and the guard that stops two models sharing one index.

The failure these prevent is silent by nature: mixed vectors write cleanly, search
successfully, and return answers drawn from a space the query does not live in.
"""

from __future__ import annotations

from typing import cast

import pytest

from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.errors import FingerprintMismatchError
from manicule.core.fingerprints import (
    PROVISIONAL_TOKENIZER_PREFIX,
    ChunkFingerprint,
    ParseFingerprint,
)


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

    parsing = cast("EmbedFingerprint", parses())
    assert not embedding.matches(parsing)
    assert not chunks().matches(cast("ChunkFingerprint", parses()))


# --- provisional counting ------------------------------------------------------------------


def test_a_stand_in_vocabulary_is_visible_in_the_fingerprint() -> None:
    """The flag is read off the identity string, so the two cannot contradict each other.

    A boolean field beside the id could say ``False`` next to a stamped one, and the ingest
    refusal would then wave through a corpus measured with an estimator.
    """
    measured = chunks(tokenizer_id="BAAI/bge-m3")
    estimated = chunks(tokenizer_id=f"{PROVISIONAL_TOKENIZER_PREFIX}x1.5:tiktoken/cl100k_base@1.0")

    assert not measured.provisional
    assert estimated.provisional
    assert not measured.matches(estimated)


def test_the_safety_factor_is_part_of_identity() -> None:
    """It multiplies every count the chunker takes, so it moves every boundary.

    Two corpora inflated by different factors are two chunkings, and a fingerprint that did
    not move would call them interchangeable.
    """
    at_one_and_a_half = chunks(tokenizer_id=f"{PROVISIONAL_TOKENIZER_PREFIX}x1.5:tiktoken@1.0")
    at_one_and_six = chunks(tokenizer_id=f"{PROVISIONAL_TOKENIZER_PREFIX}x1.6:tiktoken@1.0")

    assert at_one_and_a_half.provisional
    assert not at_one_and_a_half.matches(at_one_and_six)


# --- parse fingerprints --------------------------------------------------------------------


def parses(**overrides: object) -> ParseFingerprint:
    base: dict[str, object] = {
        "parser": "pdf",
        "version": "1",
        "libraries": {"pypdfium2": "5.12.1"},
    }
    return ParseFingerprint.model_validate({**base, **overrides})


@pytest.mark.parametrize(
    "difference",
    [
        {"parser": "plaintext"},
        {"version": "2"},
        {"libraries": {"pypdfium2": "5.13.0"}},
    ],
    ids=lambda d: next(iter(d)),
)
def test_every_field_of_a_parse_fingerprint_invalidates_a_document(
    difference: dict[str, object],
) -> None:
    """Each of the three decides what the stored text says, so each has to be identity.

    ``version`` is manicule's own extraction rules and is the one an implementation leaves
    out: a repository that changes which blocks a parser emits, without a dependency moving,
    would otherwise rewrite text under a fingerprint that stayed still.
    """
    with pytest.raises(FingerprintMismatchError):
        parses().require_match(parses(**difference))


def test_a_library_bump_invalidates_only_the_parser_that_uses_it() -> None:
    """The property the whole field exists for, stated as a comparison.

    A ``pypdfium2`` release makes every PDF stale and says nothing whatever about Markdown —
    the two fingerprints are different values and always were, so one moving cannot move the
    other.
    """
    pdf_before = parses(libraries={"pypdfium2": "5.12.1"})
    pdf_after = parses(libraries={"pypdfium2": "5.13.0"})
    markdown = parses(parser="markdown", libraries={"markdown-it-py": "4.2.0"})

    assert pdf_before.changed_fields(pdf_after) == {"libraries"}
    assert not pdf_before.matches(pdf_after)
    assert markdown.matches(parses(parser="markdown", libraries={"markdown-it-py": "4.2.0"}))


def test_a_parse_fingerprint_is_byte_stable_across_key_order() -> None:
    """It is compared as stored text, so a dictionary's insertion order must not reach it."""
    one = parses(libraries={"lxml": "6.1.1", "python-docx": "1.2.0"})
    other = parses(libraries={"python-docx": "1.2.0", "lxml": "6.1.1"})

    assert one.canonical() == other.canonical()


def test_a_parser_with_no_libraries_still_has_an_identity() -> None:
    """An empty map is a real answer, and ``version`` is then the whole of it."""
    before = parses(parser="adf", version="1", libraries={})
    after = parses(parser="adf", version="2", libraries={})

    assert not before.matches(after)
    assert "no parsing libraries" in before.describe()
