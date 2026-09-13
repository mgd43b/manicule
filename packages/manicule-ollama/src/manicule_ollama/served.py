"""What a *served* model declares about itself, read rather than assumed.

This is :mod:`manicule.embedding.cards` for a runtime that has no repository. The same
questions decide whether two sets of vectors live in the same space — the reduction, the
width, the vocabulary, the usable length — and every one of them is still read rather than
defaulted. Only the source moves: from files a model repository publishes to the GGUF metadata
the server reports and to the vectors the server actually returns.

Three of the four answers are stronger here than a repository's, and one is weaker.

**Stronger: the width is measured, not declared.** A repository is asked for ``hidden_size``;
this asks the model for a vector and counts it. The GGUF's ``embedding_length`` is read too,
and a disagreement between the two is refused rather than resolved — the vector table is
created from this number, so it has to be one number.

**Stronger: the digest is exact.** ``/api/tags`` reports the manifest digest of the blob the
server will run, so a re-pull that changes the bytes changes the identity, and vectors made by
the previous pull stop being admissible. A model name on its own could not do that.

**Stronger: the limit is the served one.** Measured on a server holding
``qwen3-embedding:0.6b``, whose GGUF declares a 32768-token context: an ``/api/embed`` with no
options was served at **4096**, and a longer input came back as a well-formed vector built from
its first 4095 tokens. So the declared context is not the limit; the limit is what this backend
asks for and then checks it got.

**Weaker: the vocabulary is configuration.** Ollama serves GGUF and exposes no tokenizer, and
manicule counts tokens to place chunk boundaries and to refuse input the model would truncate.
So ``tokenizer`` is a required setting, and deriving it from the model's name would be an
inference presented as a measurement — exactly what
:class:`~manicule.core.embedding.EmbedFingerprint` excludes ``backend`` from identity on the
understanding that nobody does. What makes it admissible instead is that the claim is
**checked**: :meth:`manicule_ollama.backend.OllamaEmbedder.setup` embeds probe strings one at a
time and compares this tokenizer's count against the server's own ``prompt_eval_count``, and a
disagreement of a single token on a single probe is a refusal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.errors import ConfigError
from manicule.embedding.cards import ModelCard
from manicule_ollama.client import (
    BACKEND,
    POOLING_TYPES,
    EmbedResult,
    OllamaClient,
    ServedModelInfo,
)
from manicule_ollama.config import OllamaEmbedderConfig

TOKENIZER_FILES: Final[tuple[str, ...]] = ("tokenizer.json",)
"""What is fetched from a tokenizer repository. One file, a few megabytes.

Narrower than :data:`manicule.embedding.cards.CARD_FILES` on purpose. A repository's
``config.json`` and ``1_Pooling/config.json`` describe *its* checkpoint, and this backend is
not running that checkpoint — it is running whatever GGUF somebody converted and pushed to
Ollama. Reading a pooling mode or a width from the repository would be reading a second model's
declaration and attributing it to this one.
"""

_PREFIX_SCHEME: Final = "prefix=none"
"""What this backend does to a text before embedding it, recorded inside the identity.

Nothing, today — and that is a statement rather than an omission. ``nomic-embed-text`` is
trained with asymmetric ``search_query:``/``search_document:`` prefixes and Qwen3-Embedding
with a query-side instruction, so applying one, or failing to, changes the vector for the same
text. manicule has no query/document distinction to hang that on:
:meth:`~manicule.core.protocols.Embedder.embed` is the only entry point and both
:mod:`manicule.ingest.embedding` and :mod:`manicule.retrieval.dense` call it the same way, so a
backend cannot tell which side it is serving. Inventing a rule here would put a retrieval
decision inside a plugin and leave it out of the fingerprint entirely.

So the honest thing is to apply nothing and to *say so in the identity*. The day manicule grows
a prefix mechanism, this term becomes the scheme's name, every stored identity stops matching,
and the corpus is re-embedded — which is the correct cost, because the vectors really would be
different. ``docs/embeddings.md`` §8 already names a prefix as something that changes the
vector for the same text; this is the field that was missing.
"""

CONTEXT_RESERVE: Final = 1
"""Total tokens held back from the served context when deriving the usable limit.

**Measured, and the two models measured disagree**, which is why it is a conservative constant
rather than arithmetic. At ``num_ctx=32768``, ``qwen3-embedding:0.6b`` accepted a 32767-token
request and refused 32768. At ``num_ctx=2048`` and at ``num_ctx=512``, ``nomic-embed-text``
accepted exactly 2048 and exactly 512. One reserves a slot and the other does not, and nothing
in either declaration says which.

Reserving one always is wrong for ``nomic-embed-text`` by a single token, in the direction that
costs nothing: a limit understated by one refuses one chunk that would have fitted, while a
limit overstated by one is a chunk indexed as its opening with no error raised. The direction
is the whole decision. :meth:`~manicule_ollama.backend.OllamaEmbedder.setup` then checks the
derived limit against the live server, so being wrong in the *other* direction — a server
serving less than this derives — is refused before a corpus is built rather than discovered in
it.
"""


@dataclass(frozen=True, slots=True)
class ServedModel:
    """One model on one server, measured, and the identity that follows from it.

    Built before the backend is usable, because the fingerprint has to exist at construction:
    the chunker takes the embedder as a construction dependency and refuses to start when its
    token budget exceeds this model's sequence limit, and that refusal has to happen before a
    corpus is built rather than after.
    """

    card: ModelCard
    """The declaration, in manicule's own vocabulary.

    A real :class:`~manicule.embedding.cards.ModelCard` rather than something card-shaped, and
    that matters beyond tidiness: :func:`manicule.ingest.workers.worker_config` reads
    ``embedder.card.path`` to find the ``tokenizer.json`` it hands to isolated parse workers,
    and a worker without one falls back to the provisional counter — whose chunks ingest
    refuses. So this attribute is load-bearing, and its ``path`` must be a directory that
    really holds that file.
    """

    info: ServedModelInfo
    """What the server said, kept beside what was derived from it."""

    num_ctx: int
    """The context this backend asks the server for, in total tokens. Sent on every request."""

    configured_name: str
    """What configuration called this model, which is **not** what identity calls it.

    The two diverge for a bare name: ``nomic-embed-text`` is served as
    ``nomic-embed-text:latest``, and :attr:`ServedModelInfo.model` carries the server's spelling
    so that one blob is one identity. The declaration cache, though, is looked up by whatever
    ``[embedding] model`` says — because that is all a metadata-only path has to go on, and it
    must not reach the server to find out what the server would call it. So the file is keyed on
    this, and its contents carry the canonical name.
    """

    weights_ref: str
    weights_identity: str

    @property
    def fingerprint(self) -> EmbedFingerprint:
        """The identity every vector this model produces is written against."""
        return self.card.fingerprint(
            backend=BACKEND, weights_ref=self.weights_ref, weights_identity=self.weights_identity
        )


class ServedDeclaration(BaseModel):
    """A :class:`ServedModel` flattened to JSON, so a later process can read it back.

    **This is the Hugging Face cache that a server does not have.** Metadata-only rebuild
    planning (:meth:`manicule.container.Container.metadata`) must derive the configured
    identity without constructing anything and without touching a network — which for the
    built-in backends means reading a model card already on disk. There is no equivalent here,
    because a served model's declaration lives in a process on another host, so this file *is*
    the on-disk copy. It is written whenever the server is read, and read-only afterwards.

    Everything a fingerprint is built from is stored, and so is the ``num_ctx`` that was asked
    for. A configuration that has since changed ``num_ctx`` would derive a different limit from
    the same server, so a record written under the old one is stale rather than usable, and is
    refused as such.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    model: str
    digest: str
    architecture: str
    context_length: int
    declared_dimension: int
    measured_dimension: int
    pooling_type: int
    """llama.cpp's raw value, **not** the reduction derived from it.

    The raw fact is what is recorded, because the derivation reads configuration too: a GGUF
    declaring no pooling takes the reduction from ``pooling``, and storing the *result* would
    leave a record that could not tell a changed setting from an unchanged one. Everything here
    is a server fact for the same reason — planning re-runs the same functions ``resolve`` does
    rather than trusting a conclusion drawn under a configuration that has since moved.
    """

    tokenizer_id: str
    special_token_count: int
    num_ctx: int
    max_sequence_length: int = Field(gt=0)
    ceiling_verified: bool = False
    """Whether a probe of exactly ``max_sequence_length`` content tokens has been accepted.

    **The one measurement here that is not remade on every start**, and the only one that costs
    a full-context forward pass: 40 seconds for a 32768-token model, measured. What it catches
    is a server whose effective ceiling sits below the ``num_ctx`` it was asked for — and the
    consequence of missing it is a *loud* failure during ingest rather than a silent one,
    because ``truncate: false`` means the server refuses an over-long input instead of
    shortening it. So the value of doing it at startup is failing before a corpus is built
    rather than during, which is worth paying for once and not worth paying for hourly.

    Every input to it is stored beside it — the digest, the served context and the limit
    derived from them — so any change at all rewrites this record and the probe runs again.
    The *free* half of the check, which is the load-bearing one, runs every time regardless:
    see :meth:`~manicule_ollama.backend.OllamaEmbedder._verify_context_limit`.
    """

    def as_info(self) -> ServedModelInfo:
        """The server facts, back in the shape the derivation functions take.

        So that planning runs the *same* functions ``resolve`` does rather than a second copy
        of the same arithmetic — which is how the two come to disagree, and the disagreement
        here is a limit that planning believes and ingest refuses.
        """
        return ServedModelInfo(
            model=self.model,
            digest=self.digest,
            architecture=self.architecture,
            context_length=self.context_length,
            embedding_length=self.declared_dimension,
            pooling_type=self.pooling_type,
            capabilities=("embedding",),
        )


def resolve(client: OllamaClient, model: str, config: OllamaEmbedderConfig) -> ServedModel:
    """Read the server, measure what it does, and settle this model's identity.

    Three round trips: ``/api/tags`` for the digest, ``/api/show`` for the architecture's
    declaration, and one ``/api/embed`` of a two-word string to measure the vector width.

    **The third one is a forward pass at construction**, which for an in-process backend would
    be the wrong shape — :class:`~manicule.embedding.base.PooledEmbedder` reads kilobytes of
    declaration at construction precisely so that gigabytes of weights can wait for ``setup``.
    The trade is different when the weights are on another host: nothing is loaded into this
    process either way, the cost is one model load in a server that was going to do it anyway,
    and the thing bought is the one number that must never be guessed. ``/api/show`` reports an
    ``embedding_length`` and it is read and cross-checked, but the vector table is created from
    what came back in an actual response.

    Raises:
        ConfigError: The server does not hold this model, its metadata is unusable, the
            configured tokenizer is missing, or the declared and measured widths disagree.
        OllamaUnavailableError: The server could not be reached.
    """
    info = client.describe(model)
    tokenizer_path, tokenizer_id = _resolve_tokenizer(config)
    specials = _special_token_count(tokenizer_path)
    pooling = _pooling(info, config.pooling)
    num_ctx = _num_ctx(info, config.num_ctx)
    usable = _usable_length(info, num_ctx, specials, config.max_sequence_length)

    probe = client.embed_sync(
        model, ["dimension probe"], num_ctx=num_ctx, keep_alive=config.keep_alive
    )
    dimension = _measured_dimension(probe, info, client.base_url)

    card = ModelCard(
        # The public model id carries no host: the same model on a second replica writes to the
        # same index, which is the whole reason a deployment address is not identity.
        model_id=f"{BACKEND}:{info.model}",
        source_ref=f"{BACKEND}:{info.model}",
        # The digest *is* the revision. It is an identity field, so a re-pull that changes the
        # bytes behind an unchanged name stops the old vectors matching — which is the one
        # thing a tag like `:latest` cannot express on its own.
        revision=f"sha256:{info.digest}",
        architecture=info.architecture,
        dimension=dimension,
        pooling=pooling,
        tokenizer_id=tokenizer_id,
        max_sequence_length=usable,
        special_token_count=specials,
        path=tokenizer_path,
    )
    return ServedModel(
        card=card,
        info=info,
        num_ctx=num_ctx,
        configured_name=model,
        weights_ref=f"{BACKEND}:{info.model}@sha256:{info.digest}",
        weights_identity=weights_identity(info.model, info.digest),
    )


def weights_identity(model: str, digest: str) -> str:
    """The stable identity of the executable artifact behind these vectors.

    Three terms, and each one is here because leaving it out would let two different vector
    spaces share an identity:

    ``ollama``
        the backend. :class:`~manicule.core.embedding.EmbedFingerprint` excludes ``backend``
        from identity *only* because this field carries the runtime boundary, and portability
        between backends is "an allowlisted measurement, not an inference from a model name".
        No measurement licenses this backend to share an identity with any other, so it does
        not: the name is in here, and nothing else can produce this string.

    the digest
        the bytes the server runs. ``ollama pull`` against a moving tag replaces them without
        changing the model's name, and a quantization change of the same model measures at
        cosine 0.92-0.97 to itself — a different space wearing one name
        (:mod:`manicule.embedding.artifacts`). The digest is what makes that a loud
        fingerprint mismatch instead of a silently mixed index.

    the prefix scheme
        what was done to the text before the model saw it. See :data:`_PREFIX_SCHEME`.
    """
    return f"artifact:{BACKEND}:{model}@sha256:{digest}:{_PREFIX_SCHEME}"


# --- persistence, for metadata-only planning ---------------------------------------------


def declaration_path(cache_dir: Path, base_url: str, model: str) -> Path:
    """Where this ``(server, model)`` pair's declaration is kept.

    Keyed by a digest of both rather than by the model's name alone: two servers can hold two
    different blobs under one name, and a file keyed on the name would let one's declaration
    describe the other's vectors. Hashed rather than spelled out because a base URL is not a
    filename.
    """
    key = hashlib.sha256(f"{base_url.rstrip('/')}\n{model}".encode()).hexdigest()[:32]
    return cache_dir / "manicule-ollama" / f"{key}.json"


def record(served: ServedModel, client: OllamaClient, cache_dir: Path) -> ServedDeclaration:
    """Flatten a measured model, and write it where planning can read it back."""
    declaration = ServedDeclaration(
        base_url=client.base_url,
        model=served.info.model,
        digest=served.info.digest,
        architecture=served.info.architecture,
        context_length=served.info.context_length,
        declared_dimension=served.info.embedding_length,
        measured_dimension=served.card.dimension,
        pooling_type=served.info.pooling_type,
        tokenizer_id=served.card.tokenizer_id,
        special_token_count=served.card.special_token_count,
        num_ctx=served.num_ctx,
        max_sequence_length=served.card.max_sequence_length,
    )
    # Keyed on the name configuration used, not the one the server answered with: planning has
    # only the former, and reaching the server to learn the latter is the thing this file exists
    # to avoid. The canonical name is inside the record.
    path = declaration_path(cache_dir, client.base_url, served.configured_name)
    previous = _read(path)
    if previous is not None and previous.model_copy(update={"ceiling_verified": False}) == (
        declaration
    ):
        # Everything this record identifies is unchanged, so a verification it already carries
        # is still a verification of *this* configuration. Dropping it here would make the
        # expensive probe run on every start, which is the cost the flag exists to avoid.
        declaration = declaration.model_copy(update={"ceiling_verified": previous.ceiling_verified})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole and replaced, never appended to: a half-written declaration read by the
    # next process would describe a model that does not exist.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(declaration.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return declaration


def _read(path: Path) -> ServedDeclaration | None:
    """A recorded declaration, or ``None`` when there is none to read.

    A record that cannot be parsed is treated as absent rather than as an error: it was written
    by an older or newer version of this package, and the worst thing it could do is be
    believed. Everything it holds is re-derivable from the server.
    """
    if not path.is_file():
        return None
    try:
        return ServedDeclaration.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ceiling_verified(cache_dir: Path, base_url: str, model: str) -> bool:
    """Whether the full-context probe has already been accepted for this exact configuration."""
    declaration = _read(declaration_path(cache_dir, base_url, model))
    return declaration is not None and declaration.ceiling_verified


def mark_ceiling_verified(cache_dir: Path, base_url: str, model: str) -> None:
    """Record that the full-context probe was accepted, so the next start need not repeat it."""
    path = declaration_path(cache_dir, base_url, model)
    declaration = _read(path)
    if declaration is None or declaration.ceiling_verified:
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        declaration.model_copy(update={"ceiling_verified": True}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def cached_fingerprint(
    cache_dir: Path, base_url: str, model: str, config: OllamaEmbedderConfig
) -> EmbedFingerprint:
    """Rebuild the configured identity from disk, with the network structurally out of reach.

    **What is stored are the server's facts, and the derivation is run again here** — through
    the same :func:`_pooling`, :func:`_num_ctx` and :func:`_usable_length` that :func:`resolve`
    uses, against the configuration currently in force. Storing the conclusions instead was
    wrong in the direction that matters: a record written before ``max_sequence_length`` was
    lowered still reported the old, larger limit, and
    ``manicule.app.runtime._rebuild_target`` compares a chunk budget against exactly that
    number — so a plan would be accepted here and refused by the live embedder partway through
    the run it authorized.

    Two things cannot be re-derived without the server or the network, and each is handled
    rather than assumed. The **width** was measured from a real vector, so the recorded one is
    used; nothing in configuration can change it. The **vocabulary** decides the special-token
    count, and reading a new one may need a download, so a ``tokenizer`` that has changed since
    the record was written is a refusal rather than a re-derivation.

    Raises:
        ConfigError: Nothing usable has been recorded for this server and model, the configured
            tokenizer is not the one the record was measured with, or the configuration in
            force is one this model cannot serve.
    """
    declaration = _read(declaration_path(cache_dir, base_url, model))
    if declaration is None:
        msg = (
            f"no usable declaration for {model!r} on {base_url} has been recorded on this "
            f"machine, so rebuild planning cannot derive the configured embedding identity "
            f"without contacting the server — and a planner that contacted it would be doing "
            f"the thing this cache exists to avoid. Run any command that builds the embedder "
            f"once (`manicule doctor` is enough) while the server is reachable."
        )
        raise ConfigError(msg)

    expected_tokenizer = tokenizer_identity(config)
    if expected_tokenizer != declaration.tokenizer_id:
        msg = (
            f"the recorded declaration for {model!r} was measured with tokenizer "
            f"{declaration.tokenizer_id!r} and configuration now names {expected_tokenizer!r}. "
            f"The vocabulary decides how many special tokens wrap every input and therefore "
            f"what the usable limit is, and reading the new one may need a download — which a "
            f"metadata-only path must not do. Run once against the server."
        )
        raise ConfigError(msg)

    info = declaration.as_info()
    num_ctx = _num_ctx(info, config.num_ctx)
    return EmbedFingerprint(
        model_id=f"{BACKEND}:{declaration.model}",
        revision=f"sha256:{declaration.digest}",
        dimension=declaration.measured_dimension,
        pooling=_pooling(info, config.pooling),
        normalized=True,
        tokenizer_id=declaration.tokenizer_id,
        max_sequence_length=_usable_length(
            info, num_ctx, declaration.special_token_count, config.max_sequence_length
        ),
        backend=BACKEND,
        weights_ref=f"{BACKEND}:{declaration.model}@sha256:{declaration.digest}",
        weights_identity=weights_identity(declaration.model, declaration.digest),
    )


# --- the individual measurements ----------------------------------------------------------


def tokenizer_identity(config: OllamaEmbedderConfig) -> str:
    """The public identity of the configured vocabulary, **without touching the network**.

    Split out of :func:`_resolve_tokenizer` so that metadata-only planning can ask whether the
    configured tokenizer is still the one a record was measured with. A local one is hashed,
    which is a local read; a remote one is named by repository and commit, which is pure.

    The identity carries no filesystem path, for the reason
    :func:`manicule.embedding.artifacts.resolve_artifact` gives about ``weights_ref``: it is
    exposed through ``index_status`` and MCP, and host directory layout is not model identity.
    A local tokenizer is identified by the digest of its bytes instead, which is also what makes
    an edited vocabulary a different identity rather than the same one.
    """
    if not config.tokenizer:
        msg = (
            "the ollama embedder needs `tokenizer` set to the repository or directory holding "
            "the tokenizer.json that matches the served model. Ollama serves GGUF and offers "
            "no tokenizer, while manicule counts tokens to place chunk boundaries and to "
            "refuse text the model would truncate without saying so. There is no default "
            "because deriving one from the model's name would be a guess recorded as an "
            "identity field — and it is checked against the server's own prompt_eval_count at "
            "setup, so naming the wrong one fails loudly rather than quietly. For "
            "`qwen3-embedding:0.6b` that is `Qwen/Qwen3-Embedding-0.6B`; for "
            "`nomic-embed-text` it is `nomic-ai/nomic-embed-text-v1.5`."
        )
        raise ConfigError(msg)

    local = Path(config.tokenizer).expanduser()
    if local.is_dir():
        if config.tokenizer_revision:
            msg = (
                f"`tokenizer` {config.tokenizer!r} is a local directory, so "
                f"`tokenizer_revision` cannot identify it. Remove it; a local vocabulary is "
                f"identified by the digest of its bytes."
            )
            raise ConfigError(msg)
        file = local / "tokenizer.json"
        if not file.is_file():
            msg = f"`tokenizer` {config.tokenizer!r} holds no tokenizer.json"
            raise ConfigError(msg)
        return f"local:sha256:{hashlib.sha256(file.read_bytes()).hexdigest()}"

    if not config.tokenizer_revision:
        msg = (
            f"`tokenizer` {config.tokenizer!r} is a repository, so `tokenizer_revision` is "
            f"required: it must be the exact 40-character commit. A branch or a tag can change "
            f"the vocabulary without changing the name, and every chunk boundary in the corpus "
            f"was measured with it — so an unpinned tokenizer is a fingerprint that does not "
            f"move when the thing it describes does."
        )
        raise ConfigError(msg)
    return f"hf:{config.tokenizer}@{config.tokenizer_revision}"


def _resolve_tokenizer(config: OllamaEmbedderConfig) -> tuple[Path, str]:
    """The directory holding ``tokenizer.json``, and the identity of that vocabulary.

    The identity comes from :func:`tokenizer_identity`, which is the half planning can compute
    without a network; this adds the half that may need one.
    """
    identity = tokenizer_identity(config)
    local = Path(config.tokenizer).expanduser()
    if local.is_dir():
        return local, identity

    from manicule.embedding.runtimes.hub import snapshot  # noqa: PLC0415 - an embeddings extra

    return snapshot(config.tokenizer, TOKENIZER_FILES, config.tokenizer_revision), identity


def _special_token_count(tokenizer_path: Path) -> int:
    """How many tokens this vocabulary wraps every input in.

    Measured by encoding the empty string, exactly as
    :meth:`manicule.embedding.runtimes.tokenization.FastTokenizer.special_token_count` does and
    for the same reason: what matters is what the tokenizer actually adds, because that is what
    eats into the sequence budget. Confirmed against the server — an empty input answered
    ``prompt_eval_count`` of 1 for ``qwen3-embedding:0.6b`` and 2 for ``nomic-embed-text``,
    matching their tokenizers exactly.
    """
    from manicule.embedding.runtimes.tokenization import FastTokenizer  # noqa: PLC0415

    return FastTokenizer(tokenizer_path / "tokenizer.json").special_token_count()


_DECLARES_NO_POOLING: Final = 0
"""llama.cpp's ``LLAMA_POOLING_TYPE_NONE``: the GGUF names no reduction."""

_POOLING: Final[dict[int, Pooling]] = {
    1: Pooling.MEAN,
    2: Pooling.CLS,
    3: Pooling.LAST_TOKEN,
}
"""llama.cpp's pooling types, mapped to the reductions manicule implements.

Written out rather than derived from the two enumerations' spellings. They agree today on mean
and CLS and disagree on the third — llama.cpp calls it ``LAST`` and manicule
:attr:`~manicule.core.embedding.Pooling.LAST_TOKEN` — and a lookup by name would have silently
stopped resolving the moment either side renamed a member, with "this model declares no
pooling" as the symptom.

:data:`~manicule_ollama.client.POOLING_TYPES` keeps llama.cpp's own names, so a value absent
from this map can be *named* in the refusal rather than reported as a number. Type 4 is
``RANK``, a reranking head rather than a reduction, and is refused for the reason
``cards.py`` refuses an unsupported Sentence-Transformers flag: taking the nearest supported
one would index a model as something it is not.
"""


def _pooling(info: ServedModelInfo, override: Pooling | None) -> Pooling:
    """The reduction the GGUF declares, in manicule's vocabulary.

    **This backend cannot pool, so it cannot be told how to.** The server reduces the token
    states and hands back a finished vector — that is what makes this tier B, "the floor rather
    than the norm". So the value here is a *record* of what happened, and configuration's role
    is limited to the one case a record can be missing: a GGUF that declares no pooling type at
    all. Even then the setting is a claim about the server rather than an instruction to it,
    and nothing verifies it; a claim that contradicts the GGUF is refused, because that setting
    would succeed and write a fingerprint saying the vectors are something they are not.
    """
    declared = _POOLING.get(info.pooling_type)
    if declared is None and info.pooling_type != _DECLARES_NO_POOLING:
        named = POOLING_TYPES.get(info.pooling_type)
        described = f"{info.pooling_type} ({named})" if named else str(info.pooling_type)
        listed = ", ".join(
            f"{value}={POOLING_TYPES.get(value, value)}" for value in sorted(_POOLING)
        )
        msg = (
            f"{info.model!r} declares {info.architecture}.pooling_type={described}, which is "
            f"not a reduction manicule knows ({listed}). Taking the nearest one would record a "
            f"fingerprint claiming a reduction the vectors did not come from."
        )
        raise ConfigError(msg)
    if declared is not None and override is not None and declared is not override:
        msg = (
            f"{info.model!r} declares {declared.value} pooling in its GGUF, but configuration "
            f"asks for {override.value}. On this backend the server does the pooling, so "
            f"`pooling` cannot change what happens — only what is recorded about it. Recording "
            f"the reduction the vectors did not come from is how an index comes to hold two "
            f"incomparable spaces under one identity. Remove the setting."
        )
        raise ConfigError(msg)
    if declared is not None:
        return declared
    if override is not None:
        return override
    msg = (
        f"{info.model!r} declares {info.architecture}.pooling_type="
        f"{_DECLARES_NO_POOLING}, meaning its GGUF names "
        f"no reduction, and configuration names none either. Pooling decides whether two sets "
        f"of vectors are comparable and cannot be guessed from a model name. Set `pooling` "
        f"under this embedder's configuration to the reduction this model was trained with, "
        f"knowing that on a served backend it is a claim about the server rather than an "
        f"instruction to it."
    )
    raise ConfigError(msg)


def _num_ctx(info: ServedModelInfo, configured: int | None) -> int:
    """The context to ask the server for, in total tokens.

    Capped by the architecture rather than merely defaulted to it. ``nomic-embed-text``'s own
    Modelfile carries ``PARAMETER num_ctx 8192`` against a GGUF declaring 2048, and the server
    serves 2048 — so a number above the architecture's is one this backend would derive a limit
    from and never receive.
    """
    if configured is None:
        return info.context_length
    if configured > info.context_length:
        msg = (
            f"`num_ctx` is {configured} but {info.model!r} declares "
            f"{info.architecture}.context_length={info.context_length}, and the server serves "
            f"the smaller of the two. Deriving a limit from the larger would claim a budget "
            f"the model never reads, and the tail of every long chunk would be dropped without "
            f"an error. Lower it, or leave it unset to use the model's own."
        )
        raise ConfigError(msg)
    return configured


def _usable_length(info: ServedModelInfo, num_ctx: int, specials: int, override: int | None) -> int:
    """Usable **content** tokens: the served context, less a reserve, less special tokens."""
    derived = num_ctx - CONTEXT_RESERVE - specials
    if derived <= 0:
        msg = (
            f"{info.model!r} served at num_ctx={num_ctx} has no room for content: a reserve of "
            f"{CONTEXT_RESERVE} and {specials} special tokens leave nothing to embed."
        )
        raise ConfigError(msg)
    if override is None:
        return derived
    if override > derived:
        msg = (
            f"`max_sequence_length` is {override} but {info.model!r} served at "
            f"num_ctx={num_ctx} reads at most {derived} content tokens "
            f"({num_ctx} - {CONTEXT_RESERVE} reserved - {specials} special). Unlike a model "
            f"repository, which may simply fail to declare its limit, this number was derived "
            f"from what the server reports — so raising it past the derivation cannot reveal "
            f"capacity, only hide truncation. Lower it, or raise `num_ctx`."
        )
        raise ConfigError(msg)
    return override


def _measured_dimension(probe: EmbedResult, info: ServedModelInfo, base_url: str) -> int:
    """The width of a vector the server actually returned, cross-checked against the GGUF."""
    if not probe.vectors:
        msg = (
            f"{base_url} answered the dimension probe for {info.model!r} with no vector. The "
            f"vector table is created from this number, so there is nothing to fall back to."
        )
        raise ConfigError(msg)
    measured = len(probe.vectors[0])
    if measured <= 0:
        msg = f"{base_url} returned an empty vector for {info.model!r}"
        raise ConfigError(msg)
    if measured != info.embedding_length:
        msg = (
            f"{info.model!r} on {base_url} declares "
            f"{info.architecture}.embedding_length={info.embedding_length} and returned a "
            f"{measured}-dimension vector. The vector table is created from this number, so it "
            f"has to be one number — and a server whose output disagrees with its own metadata "
            f"is not one whose other declarations can be taken at face value."
        )
        raise ConfigError(msg)
    return measured


__all__ = [
    "CONTEXT_RESERVE",
    "TOKENIZER_FILES",
    "ServedDeclaration",
    "ServedModel",
    "cached_fingerprint",
    "ceiling_verified",
    "declaration_path",
    "mark_ceiling_verified",
    "record",
    "resolve",
    "tokenizer_identity",
    "weights_identity",
]
