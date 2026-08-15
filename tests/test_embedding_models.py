"""What a model declares, what an artifact is allowed to be, and where both are refused.

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
from manicule.embedding.artifacts import (
    MLX_WEIGHTS,
    mlx_repo,
    mlx_weights,
    onnx_weights,
    planned_weights,
    resolve_artifact,
)
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
    """Skipping an unrecognized ``true`` indexes a max-pooled model as something else."""
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


# --- artifacts ---------------------------------------------------------------------------


def test_the_mlx_artifact_for_bge_m3_is_not_its_own_repository() -> None:
    """``BAAI/bge-m3`` publishes a PyTorch pickle and no safetensors, which MLX cannot read.

    Recorded in a table rather than discovered at load time, so the fingerprint can name the
    weights that will run before several gigabytes are fetched.
    """
    assert mlx_repo("BAAI/bge-m3") == MLX_WEIGHTS["BAAI/bge-m3"]
    assert mlx_repo("some/other-model") == "some/other-model"
    assert mlx_repo("BAAI/bge-m3", override="local/conversion") == "local/conversion"


def test_quantized_weights_are_refused(tmp_path: Path) -> None:
    """Quantization changes the vectors, and nothing downstream would catch it.

    4-bit ``bge-m3`` sits at cosine 0.92-0.97 to the same model in fp16 — a different space, not
    a rounding error. ``backend`` and ``weights_ref`` are excluded from fingerprint identity on
    the basis that a runtime does not change the output, so this is the only place the
    contradiction can be caught.
    """
    directory = write_model(tmp_path / "quantized", quantization={"bits": 4, "group_size": 64})
    (directory / "model.safetensors").write_bytes(b"not really weights")

    with pytest.raises(ConfigError, match="quantized to 4 bits"):
        mlx_weights(str(directory))


def test_weights_mlx_cannot_read_are_refused_with_the_conversion_named(tmp_path: Path) -> None:
    """The error carries the fix, because the fix is not guessable from the symptom."""
    directory = write_model(tmp_path / "no-safetensors")

    with pytest.raises(ConfigError, match="no safetensors"):
        mlx_weights(str(directory))


def test_an_unquantized_artifact_is_accepted(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "fp16")
    (directory / "model.safetensors").write_bytes(b"not really weights")

    assert mlx_weights(str(directory)).repo == str(directory)


def test_exact_weights_revision_reaches_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit recorded before setup is the commit the hub is asked to materialize."""
    revision = "1" * 40
    artifact = resolve_artifact(
        "mlx", "acme/model", "2" * 40, override="acme/conversion", revision=revision
    )
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"weights")
    seen: list[tuple[str, tuple[str, ...], str | None]] = []

    def snapshot(repo: str, patterns: tuple[str, ...], commit: str | None = None) -> Path:
        seen.append((repo, patterns, commit))
        return directory

    monkeypatch.setattr("manicule.embedding.runtimes.hub.snapshot", snapshot)

    loaded = mlx_weights(artifact)

    assert seen == [("acme/conversion", ("*.safetensors", "*.json"), revision)]
    assert loaded.describe() == f"hf:acme/conversion@{revision}"


def test_exact_onnx_revision_reaches_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "3" * 40
    artifact = resolve_artifact(
        "onnx", "acme/model", "2" * 40, override="acme/export", revision=revision
    )
    directory = tmp_path / "snapshot"
    (directory / "onnx").mkdir(parents=True)
    (directory / "onnx" / "model.onnx").write_bytes(b"graph")
    seen: list[str | None] = []

    def snapshot(repo: str, patterns: tuple[str, ...], commit: str | None = None) -> Path:
        del repo, patterns
        seen.append(commit)
        return directory

    monkeypatch.setattr("manicule.embedding.runtimes.hub.snapshot", snapshot)

    loaded, _ = onnx_weights(artifact)

    assert seen == [revision]
    assert loaded.describe() == f"hf:acme/export@{revision}"


def test_only_the_exact_allowlisted_pair_is_backend_portable() -> None:
    model = "BAAI/bge-m3"
    model_revision = "5617a9f61b028005a4858fdac845db406aefb181"
    mlx = resolve_artifact("mlx", model, model_revision)
    onnx = resolve_artifact("onnx", model, model_revision)

    assert mlx.identity == onnx.identity
    assert mlx.ref != onnx.ref

    wrong_mlx = resolve_artifact(
        "mlx",
        model,
        model_revision,
        override="mlx-community/bge-m3-mlx-fp16",
        revision="0" * 40,
    )
    wrong_onnx = resolve_artifact("onnx", model, model_revision, override=model, revision="0" * 40)
    wrong_model_revision = resolve_artifact("onnx", model, "9" * 40)
    wrong_repo = resolve_artifact(
        "onnx", model, model_revision, override="mirror/bge-m3", revision=onnx.revision or ""
    )
    assert wrong_mlx.identity != wrong_onnx.identity
    assert wrong_mlx.identity != mlx.identity
    assert wrong_model_revision.identity != onnx.identity
    assert wrong_repo.identity != onnx.identity


def test_qualified_identity_is_bound_to_both_artifact_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manicule.embedding import artifacts  # noqa: PLC0415

    model = "BAAI/bge-m3"
    model_revision = "5617a9f61b028005a4858fdac845db406aefb181"
    before = resolve_artifact("onnx", model, model_revision).identity
    monkeypatch.setitem(
        artifacts._BUILTIN_REVISIONS,  # pyright: ignore[reportPrivateUsage] - qualification seam
        (model, "mlx"),
        "f" * 40,
    )

    changed_mlx = resolve_artifact("mlx", model, model_revision)
    changed_onnx = resolve_artifact("onnx", model, model_revision)
    assert changed_mlx.identity == changed_onnx.identity
    assert changed_onnx.identity != before


def test_remote_override_without_immutable_revision_is_refused() -> None:
    with pytest.raises(ConfigError, match="40-character commit"):
        resolve_artifact("onnx", "acme/model", "1" * 40, override="acme/export")


def test_weights_revision_without_override_is_refused() -> None:
    with pytest.raises(ConfigError, match="requires an explicit"):
        resolve_artifact("onnx", "BAAI/bge-m3", "1" * 40, revision="2" * 40)


def test_weight_plan_probes_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def is_cached(repo: str, patterns: tuple[str, ...], revision: str | None = None) -> bool:
        del repo, patterns
        seen.append(revision)
        return True

    monkeypatch.setattr("manicule.embedding.runtimes.hub.is_cached", is_cached)

    plan = planned_weights("onnx", "BAAI/bge-m3")

    assert plan.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert seen == [plan.revision]


def test_local_weight_bytes_change_identity(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "local")
    weights = directory / "model.safetensors"
    weights.write_bytes(b"first")
    first = resolve_artifact("mlx", str(directory), None)
    weights.write_bytes(b"second")
    second = resolve_artifact("mlx", str(directory), None)

    assert first.ref != second.ref
    assert first.identity != second.identity


def test_local_weights_cannot_change_between_fingerprint_and_load(tmp_path: Path) -> None:
    directory = write_model(tmp_path / "local-race")
    weights = directory / "model.safetensors"
    weights.write_bytes(b"fingerprinted")
    artifact = resolve_artifact("mlx", str(directory), None)
    weights.write_bytes(b"executed instead")

    with pytest.raises(ConfigError, match="changed after their fingerprint"):
        mlx_weights(artifact)


def test_local_digest_frames_file_names_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "model.safetensors").write_bytes(b"same")
    (first / "a.json").write_bytes(b"b.jsonC")
    (second / "a.json").write_bytes(b"")
    (second / "b.json").write_bytes(b"C")

    assert (
        resolve_artifact("mlx", str(first), None).identity
        != resolve_artifact("mlx", str(second), None).identity
    )


def test_local_onnx_nested_sidecars_are_identity_bearing(tmp_path: Path) -> None:
    directory = tmp_path / "onnx-local"
    (directory / "onnx" / "data").mkdir(parents=True)
    (directory / "onnx" / "model.onnx").write_bytes(b"graph")
    sidecar = directory / "onnx" / "data" / "weights.bin"
    sidecar.write_bytes(b"first")
    first = resolve_artifact("onnx", str(directory), None)
    sidecar.write_bytes(b"second")
    second = resolve_artifact("onnx", str(directory), None)

    assert first.identity != second.identity


def test_a_repository_with_no_onnx_export_is_refused(tmp_path: Path) -> None:
    """Without an export the model runs on Apple hardware only, and says so at startup."""
    directory = write_model(tmp_path / "mlx-only")

    with pytest.raises(ConfigError, match="no ONNX export"):
        onnx_weights(str(directory))


def test_the_conventional_onnx_graph_is_preferred_over_its_quantizations(tmp_path: Path) -> None:
    """Repositories ship ``model_quantized.onnx`` beside ``model.onnx``, and they differ."""
    directory = write_model(tmp_path / "with-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model.onnx").write_bytes(b"graph")
    (directory / "onnx" / "model_quantized.onnx").write_bytes(b"graph")

    _, graph = onnx_weights(str(directory))

    assert graph.name == "model.onnx"


def test_a_lone_quantized_onnx_graph_is_refused(tmp_path: Path) -> None:
    """The case the ambiguity refusal below cannot see, and the one that would load silently.

    An ONNX graph declares no precision of its own, unlike an MLX conversion's ``config.json``,
    so a repository publishing only ``model_quantized.onnx`` is the *unambiguous* candidate. Its
    name is weak evidence and it is the only evidence there is — worth acting on, because
    admitting it puts vectors measured at cosine 0.92-0.97 to the named model into an index
    whose fingerprint says the runtime cannot have changed the output.
    """
    directory = write_model(tmp_path / "quantized-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model_quantized.onnx").write_bytes(b"graph")

    with pytest.raises(ConfigError, match="quantized"):
        onnx_weights(str(directory))


def test_a_lone_full_precision_onnx_graph_is_accepted(tmp_path: Path) -> None:
    """Half precision is not quantization: it is what the MLX side runs, and parity holds."""
    directory = write_model(tmp_path / "fp16-onnx")
    (directory / "onnx").mkdir()
    (directory / "onnx" / "model_fp16.onnx").write_bytes(b"graph")

    _, graph = onnx_weights(str(directory))

    assert graph.name == "model_fp16.onnx"


def test_several_onnx_graphs_and_no_conventional_one_is_refused(tmp_path: Path) -> None:
    """Picking the first would pick a quantization about a third of the time."""
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


def test_a_model_this_machine_does_not_have_refuses_with_the_pre_seed_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other silent download on the query path, and the one that costs 2.3 GB.

    An embedder is built and set up while a query is being answered, so a machine that has
    never held these weights fetches them *inside a search* — and on a host with no route to
    the hub, what surfaced was the hub's own exception: a repository id, a cache path, and
    nothing an operator could act on. It is not a bundle problem the way a 5 MB vocabulary is;
    nobody carries an ONNX export in a manifest, and manicule already ships a pre-seed. What
    was missing was the sentence naming it at the moment it is needed.

    Offline and synthetic: the hub is told not to look, and the repository does not exist, so
    this asserts the message rather than the network.
    """
    from manicule.embedding.runtimes.hub import (  # noqa: PLC0415 - an embeddings extra
        OFFLINE_ENV,
        ModelUnavailableError,
        snapshot,
    )

    monkeypatch.setenv(OFFLINE_ENV, "1")

    with pytest.raises(ModelUnavailableError) as raised:
        snapshot("manicule-tests/no-such-model", ["*.json"])

    message = str(raised.value)
    assert "manicule-tests/no-such-model" in message
    assert "tools/prefetch_embedding_models.py" in message
    assert OFFLINE_ENV in message
