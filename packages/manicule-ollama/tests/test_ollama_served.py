"""What the server is asked, what is derived from the answers, and what is refused.

No network: every case drives the shipped client over ``ollama_fake``'s transport, so the
request building, the JSON decoding and the error mapping under test are the real ones. That is
what lets a server's *misbehavior* be a test case — a context served below the one declared, a
width that contradicts its own metadata, a model somebody re-pulled — none of which can be
arranged on demand against a healthy host.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from manicule_ollama.client import OllamaClient
from manicule_ollama.config import OllamaEmbedderConfig
from manicule_ollama.served import (
    CONTEXT_RESERVE,
    cached_fingerprint,
    declaration_path,
    record,
    resolve,
)
from ollama_fake import ARCHITECTURE, DIGEST, MODEL, SPECIAL_TOKENS, FakeOllama, config_payload

from manicule.core.embedding import Pooling, PrefixScheme
from manicule.core.errors import ConfigError
from manicule.embedding.runtimes.hub import ModelUnavailableError
from manicule.testing import write_tokenizer


@pytest.fixture
def vocabulary(tmp_path: Path) -> Path:
    """A directory holding manicule's synthetic ``tokenizer.json``, as a local tokenizer."""
    directory = tmp_path / "vocabulary"
    directory.mkdir()
    write_tokenizer(directory / "tokenizer.json")
    return directory


def build(  # noqa: ANN201
    server: FakeOllama,
    vocabulary: Path,
    prefix_scheme: PrefixScheme = PrefixScheme.NONE,
    **overrides: object,
):
    """Resolve ``server``'s model against a local tokenizer, returning client and model."""
    config = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer=str(vocabulary), **overrides)
    )
    client = OllamaClient(config.base_url, transport=server.transport())
    return client, resolve(client, server.model, config, prefix_scheme), config


# --- identity ------------------------------------------------------------------------------


def test_the_identity_carries_the_backend_and_the_servers_own_digest(vocabulary: Path) -> None:
    """``weights_identity`` is what keeps ``backend`` safely out of the identity fields.

    `EmbedFingerprint` excludes `backend` "only because weights_identity carries the runtime
    boundary", and portability between backends is an allowlisted measurement rather than an
    inference. No measurement licenses this backend to share an identity with onnx or mlx, so
    the name is inside the string and nothing else can produce it. The digest is in there too,
    and separately in `revision`, because a re-pull is the one way these vectors change under
    an unchanged model name.

    **Two terms, and no third.** This string once ended ``:prefix=none``, a marker this backend
    kept for itself while core had no field for the question. Core owns it now, on every
    fingerprint whoever built it, so a third term here would record the same fact twice — in a
    field ``onnx`` and ``mlx`` do not write, and therefore one nothing can be compared across.
    """
    _, served, _ = build(FakeOllama(), vocabulary)

    assert served.weights_identity == f"artifact:ollama:{MODEL}@sha256:{DIGEST}"
    assert "prefix" not in served.weights_identity
    assert served.fingerprint.backend == "ollama"
    assert served.fingerprint.revision == f"sha256:{DIGEST}"


def test_the_prefix_scheme_moves_the_identity_without_touching_the_artifact(
    vocabulary: Path,
) -> None:
    """Adopting a scheme must invalidate these vectors — through core's field, not this string.

    The distinction is the whole of §9.1. Recording it in ``weights_identity`` would protect
    this backend and silently fail to protect ``onnx`` and ``mlx``, whose identities are built
    from an artifact reference that knows nothing about prefixes.
    """
    _, bare, _ = build(FakeOllama(), vocabulary)
    _, prefixed, _ = build(FakeOllama(), vocabulary, PrefixScheme.NOMIC)

    assert bare.weights_identity == prefixed.weights_identity
    assert prefixed.fingerprint.prefix_scheme is PrefixScheme.NOMIC
    assert not bare.fingerprint.matches(prefixed.fingerprint)


def test_a_document_prefix_is_charged_to_the_usable_limit(vocabulary: Path) -> None:
    """Otherwise a full chunk plus its prefix overflows the served context and is truncated.

    ``truncate: false`` makes that loud on this backend rather than silent, which is better —
    but a corpus that refuses every full-length chunk at ingest is still a corpus that cannot
    be built, and the number the chunker was handed is what decided how long chunks are.
    """
    _, bare, _ = build(FakeOllama(), vocabulary)
    _, prefixed, _ = build(FakeOllama(), vocabulary, PrefixScheme.NOMIC)

    assert prefixed.document_prefix_tokens > 0
    assert prefixed.card.max_sequence_length == (
        bare.card.max_sequence_length - prefixed.document_prefix_tokens
    )


def test_a_query_only_scheme_costs_the_document_budget_nothing(vocabulary: Path) -> None:
    """Qwen3 instructs the query and not the document, so the chunk budget must not shrink."""
    _, bare, _ = build(FakeOllama(), vocabulary)
    _, prefixed, _ = build(FakeOllama(), vocabulary, PrefixScheme.QWEN3)

    assert prefixed.document_prefix_tokens == 0
    assert prefixed.card.max_sequence_length == bare.card.max_sequence_length


def test_a_re_pull_under_the_same_name_is_a_different_vector_space(vocabulary: Path) -> None:
    """The failure a model name cannot express, and the digest can.

    `ollama pull` replaces a blob under an unchanged tag. Without the digest in identity, the
    vectors written before and after would land in one table under one fingerprint and the
    index would hold two incomparable spaces while reporting success.
    """
    _, before, _ = build(FakeOllama(), vocabulary)
    _, after, _ = build(FakeOllama(digest="b" * 64), vocabulary)

    assert not before.fingerprint.matches(after.fingerprint)
    assert before.fingerprint.canonical() != after.fingerprint.canonical()


def test_the_server_address_is_not_part_of_the_identity(vocabulary: Path) -> None:
    """A second replica of the same server writes into the same index.

    A deployment address inside an identity would mean a corpus could not be read by a copy of
    the thing that wrote it, which is a property of the deployment masquerading as a property
    of the vectors.
    """
    here = FakeOllama()
    there = FakeOllama()
    first = OllamaClient("http://one.test:11434", transport=here.transport())
    second = OllamaClient("http://two.test:11434", transport=there.transport())
    config = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(vocabulary)))

    one = resolve(first, MODEL, config)
    two = resolve(second, MODEL, config)

    assert one.fingerprint.canonical() == two.fingerprint.canonical()
    assert "one.test" not in one.fingerprint.weights_ref
    assert "one.test" not in one.fingerprint.model_id


def test_no_filesystem_path_reaches_the_public_identity(vocabulary: Path) -> None:
    """``tokenizer_id`` is exposed through index_status and MCP; host layout is not identity."""
    _, served, _ = build(FakeOllama(), vocabulary)

    assert served.fingerprint.tokenizer_id.startswith("local:sha256:")
    assert str(vocabulary) not in served.fingerprint.tokenizer_id
    assert str(vocabulary) not in str(served.fingerprint.canonical())


# --- the dimension -------------------------------------------------------------------------


def test_the_dimension_is_measured_from_a_vector_not_read_from_the_metadata(
    vocabulary: Path,
) -> None:
    """The fingerprint is the sole source of the vector width, so it comes from the model."""
    _, served, _ = build(FakeOllama(embedding_length=12), vocabulary)

    assert served.fingerprint.dimension == 12
    assert len(served.card.path.name) > 0


def test_a_width_that_contradicts_the_declaration_is_refused(vocabulary: Path) -> None:
    """A server whose output disagrees with its own metadata is refused, not reconciled.

    Either number could be the wrong one, and the vector table is created from whichever is
    chosen — so there is nothing to fall back to. It is also the clearest possible signal that
    the server's *other* declarations should not be taken at face value.
    """
    server = FakeOllama(embedding_length=8, dimension=16)

    with pytest.raises(ConfigError, match="has to be one number"):
        build(server, vocabulary)


# --- the served context --------------------------------------------------------------------


def test_the_usable_limit_is_the_served_context_less_a_reserve_and_the_specials(
    vocabulary: Path,
) -> None:
    """Usable **content** tokens, which is the unit the chunker's budget is compared in."""
    _, served, _ = build(FakeOllama(context_length=64), vocabulary)

    assert served.num_ctx == 64
    assert served.fingerprint.max_sequence_length == 64 - CONTEXT_RESERVE - SPECIAL_TOKENS


def test_num_ctx_lowers_the_limit_and_is_sent_on_every_request(vocabulary: Path) -> None:
    """The number the limit was derived from is the number the server is asked for.

    Not cosmetic: measured against a real server holding `qwen3-embedding:0.6b`, whose GGUF
    declares 32768, an /api/embed carrying no options was served at 4096 and answered a longer
    input with a vector built from its first 4095 tokens.
    """
    server = FakeOllama(context_length=64)
    _, served, _ = build(server, vocabulary, num_ctx=32)

    assert served.fingerprint.max_sequence_length == 32 - CONTEXT_RESERVE - SPECIAL_TOKENS
    embeds = [body for path, body in server.requests if path == "/api/embed"]
    assert embeds, "the dimension probe should have been sent"
    assert all(body["options"] == {"num_ctx": 32} for body in embeds)


def test_num_ctx_above_the_architecture_is_refused(vocabulary: Path) -> None:
    """A limit derived from a context the server will never grant truncates silently.

    `nomic-embed-text` ships exactly this trap in its own Modelfile: `PARAMETER num_ctx 8192`
    against a GGUF declaring 2048, and the server serves 2048.
    """
    with pytest.raises(ConfigError, match="serves the smaller of the two"):
        build(FakeOllama(context_length=2048), vocabulary, num_ctx=8192)


def test_max_sequence_length_may_be_lowered_but_not_raised(vocabulary: Path) -> None:
    """Unlike a model repository's, this number was derived from what the server reports.

    A repository may simply fail to declare its limit, which is what the override exists for
    there. Here the derivation came from the server, so raising past it cannot reveal capacity
    — only hide truncation.
    """
    _, lowered, _ = build(FakeOllama(context_length=64), vocabulary, max_sequence_length=20)
    assert lowered.fingerprint.max_sequence_length == 20

    with pytest.raises(ConfigError, match="cannot reveal capacity"):
        build(FakeOllama(context_length=64), vocabulary, max_sequence_length=1000)


# --- pooling -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pooling_type", "expected"),
    [(1, Pooling.MEAN), (2, Pooling.CLS), (3, Pooling.LAST_TOKEN)],
)
def test_pooling_is_read_from_the_gguf(
    pooling_type: int, expected: Pooling, vocabulary: Path
) -> None:
    """llama.cpp's own enumeration, translated rather than guessed from the model's name.

    Measured on the two models this backend was written for: `qwen3-embedding:0.6b` declares 3
    (last token) and `nomic-embed-text` declares 1 (mean). Nothing in either name says so.
    """
    _, served, _ = build(FakeOllama(pooling_type=pooling_type), vocabulary)

    assert served.fingerprint.pooling is expected


def test_an_unknown_pooling_type_is_refused_rather_than_approximated(vocabulary: Path) -> None:
    with pytest.raises(ConfigError, match="not a reduction manicule knows"):
        build(FakeOllama(pooling_type=97), vocabulary)


def test_a_pooling_setting_that_contradicts_the_gguf_is_refused(vocabulary: Path) -> None:
    """On a tier B backend the setting cannot change what happens — only what is recorded.

    Which makes it more dangerous than on a backend that pools, not less: it would succeed, and
    write a fingerprint claiming the vectors came from a reduction they did not.
    """
    with pytest.raises(ConfigError, match="cannot change what happens"):
        build(FakeOllama(pooling_type=1), vocabulary, pooling=Pooling.CLS)


def test_a_gguf_that_declares_no_pooling_needs_one_named(vocabulary: Path) -> None:
    with pytest.raises(ConfigError, match="names no reduction"):
        build(FakeOllama(pooling_type=0), vocabulary)

    _, served, _ = build(FakeOllama(pooling_type=0), vocabulary, pooling=Pooling.MEAN)
    assert served.fingerprint.pooling is Pooling.MEAN


# --- the tokenizer, which is configuration ---------------------------------------------------


def test_a_tokenizer_must_be_named(vocabulary: Path) -> None:
    """There is no default, because deriving one from the model name would be a guess.

    And a guess recorded in `tokenizer_id` moves every chunk boundary in the corpus while
    looking like a measurement.
    """
    server = FakeOllama()
    config = OllamaEmbedderConfig.model_validate(config_payload())
    client = OllamaClient(config.base_url, transport=server.transport())

    with pytest.raises(ConfigError, match="needs `tokenizer` set"):
        resolve(client, server.model, config)


def test_a_remote_tokenizer_must_be_pinned_to_a_commit() -> None:
    """A branch can change the vocabulary without changing the name."""
    server = FakeOllama()
    config = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer="some-org/some-tokenizer")
    )
    client = OllamaClient(config.base_url, transport=server.transport())

    with pytest.raises(ConfigError, match="`tokenizer_revision` is required"):
        resolve(client, server.model, config)


def test_a_revision_that_is_not_a_commit_is_rejected_by_configuration() -> None:
    with pytest.raises(ValueError, match="exact 40-character commit"):
        OllamaEmbedderConfig.model_validate(
            config_payload(tokenizer="some-org/some-tokenizer", tokenizer_revision="main")
        )


def test_a_local_tokenizer_cannot_carry_a_revision(vocabulary: Path) -> None:
    with pytest.raises(ConfigError, match="local directory"):
        build(FakeOllama(), vocabulary, tokenizer_revision="a" * 40)


def test_weights_settings_are_refused_rather_than_ignored() -> None:
    """The weights are on the other side of an HTTP connection; nothing here could act on them.

    `extra="forbid"` on `EmbedderConfig` is doing real work, and a field accepted and then
    ignored is the same defect with the sign flipped.
    """
    with pytest.raises(ValueError, match="cannot honor `weights`"):
        OllamaEmbedderConfig.model_validate(config_payload(weights="some-org/some-weights"))


# --- what the server has to be --------------------------------------------------------------


def test_a_model_without_the_embedding_capability_is_refused(vocabulary: Path) -> None:
    """Ollama answers /api/embed for a generative model by pooling its hidden states.

    Which produces correctly shaped, correctly normalized vectors from a model never trained to
    make them — the failure with no symptom other than worse retrieval.
    """
    server = FakeOllama(capabilities=("completion", "tools"))

    with pytest.raises(ConfigError, match="does not declare the 'embedding' capability"):
        build(server, vocabulary)


def test_a_model_the_server_does_not_hold_names_what_it_does(vocabulary: Path) -> None:
    server = FakeOllama()
    config = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(vocabulary)))
    client = OllamaClient(config.base_url, transport=server.transport())

    with pytest.raises(ConfigError, match="It holds: "):
        resolve(client, "not-pulled:latest", config)


def test_the_servers_spelling_of_a_name_is_the_one_that_reaches_identity(
    vocabulary: Path,
) -> None:
    """`ollama list` shows `nomic-embed-text:latest` for what configuration calls
    `nomic-embed-text`, and one blob has to be one identity.

    Being told a model visible in `ollama list` does not exist would be wrong, and so would the
    other half: if the configured spelling reached the fingerprint, an operator who rewrote
    their configuration to the tag Ollama itself prints would get a mismatch against their own
    index and a full re-embed, for a change that moved nothing about the vectors.
    """
    server = FakeOllama(model="synthetic-embed:latest")
    config = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(vocabulary)))

    bare = resolve(
        OllamaClient(config.base_url, transport=server.transport()), "synthetic-embed", config
    )
    tagged = resolve(
        OllamaClient(config.base_url, transport=server.transport()),
        "synthetic-embed:latest",
        config,
    )

    assert bare.info.digest == tagged.info.digest == DIGEST
    assert bare.fingerprint.model_id == "ollama:synthetic-embed:latest"
    assert bare.fingerprint.canonical() == tagged.fingerprint.canonical()
    assert bare.weights_identity == tagged.weights_identity


def test_missing_gguf_metadata_is_refused_by_name(vocabulary: Path) -> None:
    """A context length or a width that is absent is never substituted."""
    server = FakeOllama(architecture=ARCHITECTURE, context_length=0)

    with pytest.raises(ConfigError, match="context_length"):
        build(server, vocabulary)


# --- the declaration cache -------------------------------------------------------------------


def test_the_recorded_declaration_rebuilds_the_same_fingerprint_offline(
    vocabulary: Path, tmp_path: Path
) -> None:
    """Metadata-only rebuild planning has no server to ask, and no model card to read.

    So the declaration written when the server *was* read is the equivalent of the Hugging Face
    cache the built-in backends plan from. It has to reproduce the identity exactly, because a
    planner deriving a different one would rebuild a corpus that did not need rebuilding.
    """
    client, served, config = build(FakeOllama(), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)

    replayed = cached_fingerprint(cache, client.base_url, MODEL, config)

    assert replayed.canonical() == served.fingerprint.canonical()
    assert replayed.weights_identity == served.weights_identity
    assert replayed.max_sequence_length == served.fingerprint.max_sequence_length


def test_a_prefixed_declaration_replays_its_budget_without_the_vocabulary(
    vocabulary: Path, tmp_path: Path
) -> None:
    """Counting a prefix needs the tokenizer, which a metadata-only path may not read.

    So the cost is recorded beside the special-token count and replayed from the record, the
    same way. Recomputing it here would either contact a repository or guess, and a planner
    that guessed the budget would authorize a rebuild the live embedder then refuses partway.
    """
    client, served, config = build(FakeOllama(), vocabulary, PrefixScheme.NOMIC)
    cache = tmp_path / "cache"
    record(served, client, cache)

    replayed = cached_fingerprint(cache, client.base_url, MODEL, config, PrefixScheme.NOMIC)

    assert replayed.prefix_scheme is PrefixScheme.NOMIC
    assert replayed.canonical() == served.fingerprint.canonical()
    assert replayed.max_sequence_length == served.fingerprint.max_sequence_length


def test_a_prefix_scheme_that_has_moved_since_the_record_is_refused(
    vocabulary: Path, tmp_path: Path
) -> None:
    """A refusal rather than a re-derivation, for the reason a changed tokenizer is.

    Deriving the new budget means counting the new prefix under this model's vocabulary, and
    reading a vocabulary may need a download — which is the thing a metadata-only path exists
    to avoid. Silently replaying the old budget would be worse than either: the planner would
    believe a limit the live embedder no longer has.
    """
    client, served, config = build(FakeOllama(), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)

    with pytest.raises(ConfigError, match="prefix scheme"):
        cached_fingerprint(cache, client.base_url, MODEL, config, PrefixScheme.NOMIC)


def test_planning_refuses_rather_than_contacting_the_server(
    vocabulary: Path, tmp_path: Path
) -> None:
    """A planner that reached the network would be doing the thing this path exists to avoid."""
    config = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(vocabulary)))

    with pytest.raises(ConfigError, match="has been recorded on this machine"):
        cached_fingerprint(tmp_path / "cache", config.base_url, MODEL, config)


def test_planning_re_derives_under_the_configuration_in_force(
    vocabulary: Path, tmp_path: Path
) -> None:
    """A record is server *facts*, and the derivation is run again against current settings.

    Storing the conclusions was wrong in the one direction that matters. A declaration written
    before `max_sequence_length` was lowered still reported the old, larger limit — and
    `manicule.app.runtime._rebuild_target` compares a chunk budget against exactly that number,
    so a plan would be accepted here and refused by the live embedder partway through the run
    it had authorized.
    """
    client, served, _ = build(FakeOllama(context_length=64), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)
    assert served.fingerprint.max_sequence_length == 64 - CONTEXT_RESERVE - SPECIAL_TOKENS

    for overrides, expected in (
        ({"num_ctx": 32}, 32 - CONTEXT_RESERVE - SPECIAL_TOKENS),
        ({"max_sequence_length": 20}, 20),
    ):
        changed = OllamaEmbedderConfig.model_validate(
            config_payload(tokenizer=str(vocabulary), **overrides)
        )
        live = resolve(
            OllamaClient(changed.base_url, transport=FakeOllama(context_length=64).transport()),
            MODEL,
            changed,
        )
        planned = cached_fingerprint(cache, client.base_url, MODEL, changed)

        assert planned.max_sequence_length == expected
        assert planned.canonical() == live.fingerprint.canonical()


def test_planning_refuses_a_configuration_the_model_cannot_serve(
    vocabulary: Path, tmp_path: Path
) -> None:
    """The same derivation means the same refusals, offline.

    A `num_ctx` above the architecture's is a limit the server would never grant, and it is
    refused here for the reason it is refused live rather than quietly clamped into a number
    that reads as a measurement.
    """
    client, served, _ = build(FakeOllama(context_length=2048), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)
    changed = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer=str(vocabulary), num_ctx=8192)
    )

    with pytest.raises(ConfigError, match="serves the smaller of the two"):
        cached_fingerprint(cache, client.base_url, MODEL, changed)


def test_planning_refuses_a_record_measured_with_another_vocabulary(
    vocabulary: Path, tmp_path: Path
) -> None:
    """The one input planning cannot re-derive, because reading it may need a download.

    The vocabulary decides how many special tokens wrap every input and therefore what the
    usable limit is. A metadata-only path must not fetch one, so a changed `tokenizer` is a
    refusal naming the command that would rewrite the record.
    """
    client, served, _ = build(FakeOllama(), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)
    other = tmp_path / "other"
    other.mkdir()
    write_tokenizer(other / "tokenizer.json")
    (other / "tokenizer.json").write_bytes(
        (other / "tokenizer.json").read_bytes() + b"\n"  # same vocabulary, different bytes
    )
    changed = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(other)))

    with pytest.raises(ConfigError, match="was measured with tokenizer"):
        cached_fingerprint(cache, client.base_url, MODEL, changed)


def test_an_unreadable_record_is_absent_rather_than_an_exception(
    vocabulary: Path, tmp_path: Path
) -> None:
    """A half-written or schema-shifted record must not escape as a library traceback.

    Every value in it is re-derivable from the server, so the useful answer is the one that
    names the command which would rewrite it — not a validation error from inside pydantic,
    arriving in the middle of a rebuild plan.
    """
    client, served, config = build(FakeOllama(), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)
    declaration_path(cache, client.base_url, MODEL).write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="has been recorded on this machine"):
        cached_fingerprint(cache, client.base_url, MODEL, config)


def test_a_bare_name_records_and_plans_under_one_key(vocabulary: Path, tmp_path: Path) -> None:
    """The lookup key is the configured name; the canonical one lives inside the record.

    These are two different names for a bare configuration — `nomic-embed-text` is served as
    `nomic-embed-text:latest` — and they are needed for two different things. Identity has to
    use the server's, so one blob is one vector space. The declaration *cache* has to use
    configuration's, because a metadata-only path has nothing else to look one up by and must
    not reach the server to find out what the server would call it.

    Writing under one and reading under the other left rebuild planning reporting "nothing has
    been recorded on this machine" against a record sitting on disk, for every model configured
    without a tag. Found by running the real container rather than by reading.
    """
    server = FakeOllama(model="synthetic-embed:latest")
    config = OllamaEmbedderConfig.model_validate(config_payload(tokenizer=str(vocabulary)))
    client = OllamaClient(config.base_url, transport=server.transport())
    served = resolve(client, "synthetic-embed", config)
    cache = tmp_path / "cache"
    record(served, client, cache)

    planned = cached_fingerprint(cache, client.base_url, "synthetic-embed", config)

    assert planned.canonical() == served.fingerprint.canonical()
    assert planned.model_id == "ollama:synthetic-embed:latest"


def test_two_writers_do_not_share_one_temporary_file(vocabulary: Path, tmp_path: Path) -> None:
    """The cache directory is shared, so the scratch name cannot be.

    A manicule server and an ingest run against one data directory both write this record. With
    a fixed ``<name>.json.tmp`` beside it, the second writer opens the first writer's in-flight
    file, overwrites its bytes and then renames it into place — so what lands is a declaration
    neither process wrote, read by the next one as fact.

    Simulated by leaving that exact deterministic name in place as another writer's half-written
    file: this process must neither read it, overwrite it, nor rename it away.
    """
    client, served, config = build(FakeOllama(), vocabulary)
    cache = tmp_path / "cache"
    record(served, client, cache)
    path = declaration_path(cache, client.base_url, MODEL)

    shared = path.with_suffix(".json.tmp")
    shared.write_text("{half written by another process", encoding="utf-8")
    record(served, client, cache)

    assert shared.is_file(), (
        "another writer's in-flight file was renamed away, so both processes were using one "
        "scratch name and each could publish the other's partial bytes"
    )
    assert shared.read_text(encoding="utf-8") == "{half written by another process"
    assert cached_fingerprint(cache, client.base_url, MODEL, config).canonical() == (
        served.fingerprint.canonical()
    )


def test_a_tokenizer_that_is_not_on_this_machine_names_this_backends_own_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal an operator actually hits in a container, and it used to misdirect them.

    ``snapshot`` defaults to suggesting ``embedding.model`` or a backend's ``weights``. Here
    both are wrong: ``embedding.model`` names a model on the *server*, and ``weights`` is
    refused outright by this backend's own validator — so an operator who followed the default
    advice would be told their configuration was invalid by the very next error. What is
    missing is a ``tokenizer.json``, and ``tokenizer`` is the setting that supplies it.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-cache"))
    config = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer="Qwen/Qwen3-Embedding-0.6B", tokenizer_revision="a" * 40)
    )
    client = OllamaClient(config.base_url, transport=FakeOllama().transport())

    with pytest.raises(ModelUnavailableError) as refusal:
        resolve(client, MODEL, config)

    message = str(refusal.value)
    assert "tokenizer" in message
    assert "--backend ollama" in message
    assert "`weights`" not in message
    assert "embedding.model" not in message
