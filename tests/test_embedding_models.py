"""What a model declares, what an artefact is allowed to be, and where both are refused.

Every case here is a synthetic repository written in the test, so the whole file runs offline
in under a second and can express declarations no published model ships: two pooling flags at
once, a dimension that disagrees with itself, a sequence length that is simply absent. Those
are the interesting ones, because they are where a default would take over.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from manicule.config.settings import ENV_PREFIX
from manicule.core.embedding import Pooling
from manicule.core.errors import ConfigError
from manicule.embedding.artifacts import MLX_WEIGHTS, mlx_repo, mlx_weights, onnx_weights
from manicule.embedding.cards import load_tokenizer, read_card
from tests.embedding_support import (
    REQUIRE_MODELS_ENV,
    REQUIRED_MODELS,
    write_model,
    write_tokenizer,
)


def test_the_models_required_switch_survives_the_test_environment() -> None:
    """The switch that stops a green job from meaning nothing, and it silently did not work.

    ``manicule_environment`` deletes every ``MANICULE_``-prefixed variable before each test, so
    that a developer's own configuration cannot leak into the suite. The first version of this
    switch was called ``MANICULE_REQUIRE_EMBEDDING_MODELS`` and lived inside that namespace, so
    it was deleted before it was ever read: CI set it, the backend suites skipped every case for
    which no weights were present, and the job reported success — the exact failure the switch
    exists to prevent, inside the mechanism meant to prevent it. Found by reading a green CI log
    rather than by any test, which is why there is now a test.

    Two things are asserted, because either alone would have let it through: the name is outside
    manicule's namespace, and what the environment holds *during a test* is what was read at
    import time.
    """
    assert not REQUIRE_MODELS_ENV.startswith(ENV_PREFIX), (
        f"{REQUIRE_MODELS_ENV} is inside manicule's configuration namespace, which the test "
        f"environment fixture clears before every test. It configures the suite, not the "
        f"application, and a switch that is deleted before it is read disables itself in silence"
    )

    live = os.environ.get(REQUIRE_MODELS_ENV, "")
    expected = frozenset(name.strip() for name in live.split(",") if name.strip())
    assert expected == REQUIRED_MODELS


def test_a_model_is_read_rather_than_guessed(tmp_path: Path) -> None:
    """Reduction, width, vocabulary and usable length all come from the repository's own files."""
    directory = write_model(tmp_path / "model")

    card = read_card(str(directory))

    assert card.pooling is Pooling.CLS
    assert card.dimension == 8
    assert card.architecture == "xlm-roberta"
    assert card.special_token_count == 2


def test_usable_length_is_content_tokens_not_the_positional_limit(tmp_path: Path) -> None:
    """The number the chunk budget is compared against has to be the number that truncates.

    ``max_position_embeddings`` is not it. For BGE-M3 the config says 8194, XLM-RoBERTa's
    position ids start at ``pad_token_id + 1`` so 8192 are addressable, and two of those go to
    ``<s>`` and ``</s>`` — 8190 usable. Getting this wrong in the generous direction drops the
    tail of every full chunk with no error raised.
    """
    directory = write_model(tmp_path / "model", max_seq_length=32, max_position_embeddings=64)

    card = read_card(str(directory))

    assert card.max_sequence_length == 30


def test_the_declared_length_wins_when_it_is_lower_than_the_architecture_allows(
    tmp_path: Path,
) -> None:
    """Repositories routinely ship configured far below their positional capacity."""
    directory = write_model(tmp_path / "model", max_seq_length=16, max_position_embeddings=8192)

    assert read_card(str(directory)).max_sequence_length == 14


def test_the_positional_capacity_wins_when_the_declaration_overreaches(tmp_path: Path) -> None:
    """A repository claiming more than the embeddings can address would truncate in silence.

    64 positions, less the RoBERTa-family offset of ``pad_token_id + 1`` — 3 here — less two
    special tokens, is 59. The offset is read from the config rather than hardcoded at 2,
    because it is the padding index that decides it.
    """
    directory = write_model(tmp_path / "model", max_seq_length=4096, max_position_embeddings=64)

    assert read_card(str(directory)).max_sequence_length == 59


def test_a_bert_style_model_has_no_position_offset(tmp_path: Path) -> None:
    """The offset is a RoBERTa-family convention, not a universal one."""
    directory = write_model(
        tmp_path / "model", model_type="bert", max_seq_length=64, max_position_embeddings=64
    )

    assert read_card(str(directory)).max_sequence_length == 62


def test_a_model_declaring_no_pooling_is_refused(tmp_path: Path) -> None:
    """Refused, not defaulted.

    Mean is the obvious default and is wrong for every CLS-pooled model — which includes the
    one manicule ships with. A default here would be a guess wearing a measurement's clothes.
    """
    directory = write_model(tmp_path / "model", pooling_file=False)

    with pytest.raises(ConfigError, match="declares no pooling"):
        read_card(str(directory))

    assert read_card(str(directory), pooling_override=Pooling.MEAN).pooling is Pooling.MEAN


def test_configuration_may_not_contradict_a_model_s_declared_pooling(tmp_path: Path) -> None:
    """A setting that overrules how the weights were trained succeeds, which is the problem."""
    directory = write_model(tmp_path / "model", pooling_flags={"pooling_mode_cls_token": True})

    with pytest.raises(ConfigError, match="declares cls pooling"):
        read_card(str(directory), pooling_override=Pooling.MEAN)


def test_an_unimplemented_pooling_mode_is_named_rather_than_skipped(tmp_path: Path) -> None:
    """Skipping an unrecognised ``true`` indexes a max-pooled model as something else."""
    directory = write_model(
        tmp_path / "model",
        pooling_flags={"pooling_mode_max_tokens": True, "pooling_mode_mean_tokens": True},
    )

    with pytest.raises(ConfigError, match="pooling_mode_max_tokens"):
        read_card(str(directory))


def test_two_declared_reductions_are_refused(tmp_path: Path) -> None:
    """Sentence-Transformers concatenates them into a wider vector; manicule does not."""
    directory = write_model(
        tmp_path / "model",
        pooling_flags={"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True},
    )

    with pytest.raises(ConfigError, match="more than one pooling mode"):
        read_card(str(directory))


def test_a_disagreement_about_the_dimension_is_refused(tmp_path: Path) -> None:
    """The vector table is created from this number, so it has to be one number."""
    directory = write_model(tmp_path / "model", hidden_size=8, word_embedding_dimension=16)

    with pytest.raises(ConfigError, match="word_embedding_dimension"):
        read_card(str(directory))


def test_a_model_declaring_no_dimension_is_refused(tmp_path: Path) -> None:
    """Never assumed: a wrong dimension builds an index that accepts every write."""
    directory = write_model(tmp_path / "model", hidden_size=None, word_embedding_dimension=None)

    with pytest.raises(ConfigError, match="no vector width"):
        read_card(str(directory))


def test_a_model_declaring_no_sequence_length_is_refused(tmp_path: Path) -> None:
    """ "Unknown" is exactly the state that produces silent truncation, so it is not expressible."""
    directory = write_model(tmp_path / "model", max_seq_length=None)

    with pytest.raises(ConfigError, match="no max_seq_length"):
        read_card(str(directory))

    assert read_card(str(directory), max_sequence_length_override=100).max_sequence_length == 100


def test_a_model_without_a_fast_tokenizer_is_refused(tmp_path: Path) -> None:
    """Token counts are measured with the model's own vocabulary or not at all.

    A stand-in vocabulary undercounts, and undercounting is the direction that truncates.
    """
    directory = write_model(tmp_path / "model", tokenizer=False)

    with pytest.raises(ConfigError, match=r"tokenizer\.json"):
        read_card(str(directory))


def test_a_local_model_records_no_revision(tmp_path: Path) -> None:
    """There is no commit to pin, and inventing one is a fingerprint claiming a guarantee."""
    card = read_card(str(write_model(tmp_path / "model")))

    assert card.revision is None
    assert card.fingerprint(backend="stub").revision is None


def test_the_tokenizer_pads_with_the_model_s_own_pad_token(tmp_path: Path) -> None:
    """Which token pads is decisive for an unmasked reduction, so it is read rather than assumed."""
    card = read_card(str(write_model(tmp_path / "model")))

    encoded = load_tokenizer(card).encode_batch(["alpha", "alpha beta gamma"])

    assert len(encoded.ids[0]) == len(encoded.ids[1])
    assert encoded.attention_mask[0] == [1, 1, 1, 0, 0]
    assert encoded.ids[0][-1] == 2, "padding must use <pad>, not whatever id came first"


def test_a_model_with_no_declared_pad_token_is_refused(tmp_path: Path) -> None:
    """Batches cannot be assembled without one, and picking one would be inventing a vocabulary."""
    directory = write_model(tmp_path / "model")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "config.json").write_text(
        '{"model_type": "xlm-roberta", "hidden_size": 8, "max_position_embeddings": 64}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="no padding token"):
        load_tokenizer(read_card(str(directory)))


# --- artefacts ---------------------------------------------------------------------------


def test_the_mlx_artefact_for_bge_m3_is_not_its_own_repository() -> None:
    """``BAAI/bge-m3`` publishes a PyTorch pickle and no safetensors, which MLX cannot read.

    Recorded in a table rather than discovered at load time, so the fingerprint can name the
    weights that will run before several gigabytes are fetched.
    """
    assert mlx_repo("BAAI/bge-m3") == MLX_WEIGHTS["BAAI/bge-m3"]
    assert mlx_repo("some/other-model") == "some/other-model"
    assert mlx_repo("BAAI/bge-m3", override="local/conversion") == "local/conversion"


def test_quantised_weights_are_refused(tmp_path: Path) -> None:
    """Quantisation changes the vectors, and nothing downstream would catch it.

    4-bit ``bge-m3`` sits at cosine 0.92-0.97 to the same model in fp16 — a different space, not
    a rounding error. ``backend`` and ``weights_ref`` are excluded from fingerprint identity on
    the basis that a runtime does not change the output, so this is the only place the
    contradiction can be caught.
    """
    directory = write_model(tmp_path / "quantised", quantization={"bits": 4, "group_size": 64})
    (directory / "model.safetensors").write_bytes(b"not really weights")

    with pytest.raises(ConfigError, match="quantised to 4 bits"):
        mlx_weights(str(directory))


def test_weights_mlx_cannot_read_are_refused_with_the_conversion_named(tmp_path: Path) -> None:
    """The error carries the fix, because the fix is not guessable from the symptom."""
    directory = write_model(tmp_path / "no-safetensors")

    with pytest.raises(ConfigError, match="no safetensors"):
        mlx_weights(str(directory))


def test_an_unquantised_artefact_is_accepted(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "fp16")
    (directory / "model.safetensors").write_bytes(b"not really weights")

    assert mlx_weights(str(directory)).repo == str(directory)


def test_a_repository_with_no_onnx_export_is_refused(tmp_path: Path) -> None:
    """Without an export the model runs on Apple hardware only, and says so at startup."""
    directory = write_model(tmp_path / "mlx-only")

    with pytest.raises(ConfigError, match="no ONNX export"):
        onnx_weights(str(directory))


def test_the_conventional_onnx_graph_is_preferred_over_its_quantisations(tmp_path: Path) -> None:
    """Repositories ship ``model_quantized.onnx`` beside ``model.onnx``, and they differ."""
    directory = write_model(tmp_path / "with-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model.onnx").write_bytes(b"graph")
    (directory / "onnx" / "model_quantized.onnx").write_bytes(b"graph")

    _, graph = onnx_weights(str(directory))

    assert graph.name == "model.onnx"


def test_a_lone_quantised_onnx_graph_is_refused(tmp_path: Path) -> None:
    """The case the ambiguity refusal below cannot see, and the one that would load silently.

    An ONNX graph declares no precision of its own, unlike an MLX conversion's ``config.json``,
    so a repository publishing only ``model_quantized.onnx`` is the *unambiguous* candidate. Its
    name is weak evidence and it is the only evidence there is — worth acting on, because
    admitting it puts vectors measured at cosine 0.92-0.97 to the named model into an index
    whose fingerprint says the runtime cannot have changed the output.
    """
    directory = write_model(tmp_path / "quantised-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model_quantized.onnx").write_bytes(b"graph")

    with pytest.raises(ConfigError, match="quantised"):
        onnx_weights(str(directory))


def test_a_lone_full_precision_onnx_graph_is_accepted(tmp_path: Path) -> None:
    """Half precision is not quantisation: it is what the MLX side runs, and parity holds."""
    directory = write_model(tmp_path / "fp16-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model_fp16.onnx").write_bytes(b"graph")

    _, graph = onnx_weights(str(directory))

    assert graph.name == "model_fp16.onnx"


def test_several_onnx_graphs_and_no_conventional_one_is_refused(tmp_path: Path) -> None:
    """Picking the first would pick a quantisation about a third of the time."""
    directory = write_model(tmp_path / "ambiguous")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model_fp16.onnx").write_bytes(b"graph")
    (directory / "onnx" / "model_quantized.onnx").write_bytes(b"graph")

    with pytest.raises(ConfigError, match="2 ONNX graphs"):
        onnx_weights(str(directory))


def test_the_synthetic_tokenizer_wraps_input_the_way_xlm_roberta_does(tmp_path: Path) -> None:
    """Guards the fixture itself: a fake with the wrong special tokens would move every number."""
    from manicule.embedding.runtimes.tokenization import FastTokenizer  # noqa: PLC0415

    write_tokenizer(tmp_path / "tokenizer.json")
    tokenizer = FastTokenizer(tmp_path / "tokenizer.json")

    assert tokenizer.special_token_count() == 2
    assert tokenizer.content_ids("alpha beta") == [4, 5]
