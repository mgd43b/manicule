"""What a model declares about itself, read rather than assumed.

Everything that decides whether two sets of vectors live in the same space — the reduction,
the width, the vocabulary, the usable length — comes from files the model repository
publishes. Nothing here has a default, and that is the point: a default is a value that looks
like a measurement and is not, and the failure it produces is an index that works.

The declaration is read from the **canonical** repository even when the weights are executed
from somewhere else. Conversions drop things: ``mlx-community/bge-m3-mlx-fp16`` ships no
``1_Pooling/config.json`` at all, so a backend that read its pooling from the artifact it
loaded would find nothing and fall back — to mean, which is wrong for this model. Identity
belongs to the model; the artifact is an implementation detail of running it
(:mod:`manicule.embedding.artifacts`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from manicule.embedding.runtimes.tokenization import FastTokenizer

CARD_FILES: Final = (
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
"""The declaration, without the weights. A few kilobytes, fetched even for a local backend."""

_POOLING_FLAGS: Final[dict[str, Pooling]] = {
    "pooling_mode_cls_token": Pooling.CLS,
    "pooling_mode_mean_tokens": Pooling.MEAN,
    "pooling_mode_lasttoken": Pooling.LAST_TOKEN,
}
"""Sentence-Transformers pooling flags manicule implements."""

_UNSUPPORTED_POOLING_FLAGS: Final[tuple[str, ...]] = (
    "pooling_mode_max_tokens",
    "pooling_mode_mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens",
)
"""Declared reductions manicule does not implement.

Named rather than ignored. Skipping an unrecognized ``true`` and taking the next flag that
happens to be set is how a max-pooled model gets indexed as a mean-pooled one.
"""

_ROBERTA_FAMILY: Final[frozenset[str]] = frozenset({"xlm-roberta", "roberta", "camembert"})
"""Architectures whose position ids start at ``pad_token_id + 1``.

``max_position_embeddings`` therefore overstates the usable length by that offset: BGE-M3
declares 8194 and attends to 8192. Getting this wrong in the generous direction truncates.
"""


class ModelCard(BaseModel):
    """A model's own declaration of what its vectors are.

    Built once at startup and turned into an :class:`~manicule.core.embedding.EmbedFingerprint`
    that every later write is compared against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str | None = Field(
        default=None,
        description="The resolved hub commit or local declaration/tokenizer digest, so model "
        "inputs cannot change under a corpus without the fingerprint changing.",
    )
    architecture: str = Field(
        min_length=1,
        description="``config.json``'s ``model_type``. Recorded because it decides which "
        "tensor a backend binds to which name.",
    )
    dimension: int = Field(gt=0)
    pooling: Pooling
    tokenizer_id: str = Field(min_length=1)
    max_sequence_length: int = Field(
        gt=0,
        description="**Usable content tokens**: the declared sequence length, capped by what "
        "the position embeddings can address, minus the special tokens the tokenizer adds. "
        "Not ``max_position_embeddings``, which for BGE-M3 is 8194 against 8190 usable.",
    )
    special_token_count: int = Field(
        ge=0,
        description="How many tokens the tokenizer wraps every input in. Measured by encoding "
        "the empty string, not counted off a list of names.",
    )
    path: Path = Field(description="Local directory holding the declaration files.")

    def fingerprint(
        self, *, backend: str, weights_ref: str = "", weights_identity: str = ""
    ) -> EmbedFingerprint:
        """The identity every vector this model produces is written against.

        ``normalized`` is always ``True``: normalization is applied in
        :mod:`manicule.embedding.pooling` rather than read from the model's declared pipeline.
        """
        return EmbedFingerprint(
            model_id=self.model_id,
            revision=self.revision,
            dimension=self.dimension,
            pooling=self.pooling,
            normalized=True,
            tokenizer_id=self.tokenizer_id,
            max_sequence_length=self.max_sequence_length,
            backend=backend,
            weights_ref=weights_ref,
            weights_identity=weights_identity,
        )


def read_card(
    model_id: str,
    *,
    revision: str | None = None,
    pooling_override: Pooling | None = None,
    max_sequence_length_override: int | None = None,
) -> ModelCard:
    """Read a model's declaration, downloading only the metadata files.

    Args:
        model_id: A Hugging Face repository id, or a path to a local directory.
        revision: A commit, branch or tag. Resolved to a commit and recorded.
        pooling_override: Used **only** when the repository declares no pooling. Supplying one
            that contradicts the repository is refused rather than obeyed.
        max_sequence_length_override: Likewise, and in the same units as
            :attr:`ModelCard.max_sequence_length` — usable content tokens.

    Raises:
        ConfigError: The repository declares nothing usable and configuration supplied
            nothing either, or the two disagree.
    """
    path, resolved = _fetch(model_id, revision)
    config = read_json(path / "config.json", model_id)
    pooling_config = read_json_if_present(path / "1_Pooling" / "config.json")

    pooling = _resolve_pooling(model_id, pooling_config, pooling_override)
    dimension = _resolve_dimension(model_id, config, pooling_config)
    tokenizer = _load_tokenizer(path, model_id)
    specials = tokenizer.special_token_count()
    usable = _resolve_length(model_id, path, config, specials, max_sequence_length_override)

    return ModelCard(
        model_id=model_id,
        revision=resolved,
        architecture=str(config.get("model_type") or "unknown"),
        dimension=dimension,
        pooling=pooling,
        tokenizer_id=model_id,
        max_sequence_length=usable,
        special_token_count=specials,
        path=path,
    )


def load_tokenizer(card: ModelCard) -> FastTokenizer:
    """The model's own tokenizer, padded with the model's own pad token.

    Truncation stays off — see :mod:`manicule.embedding.runtimes.tokenization` — so an
    over-long input arrives at full length and is refused by name rather than shortened into a
    vector that describes an opening fragment.
    """
    tokenizer = _load_tokenizer(card.path, card.model_id)
    pad_id, pad_token = _padding(card, tokenizer)
    tokenizer.enable_padding(pad_id, pad_token)
    return tokenizer


def _padding(card: ModelCard, tokenizer: FastTokenizer) -> tuple[int, str]:
    """The pad token, from the tokenizer's own configuration.

    Which token pads is irrelevant to a masked reduction and decisive for an unmasked one, so
    it is read rather than assumed — and an id with no token, or a token with no id, is a
    broken repository rather than something to paper over.
    """
    config = read_json_if_present(card.path / "tokenizer_config.json")
    declared: object = config.get("pad_token")
    # Repositories write this either as a bare string or as a full AddedToken object.
    if isinstance(declared, dict):
        declared = cast("dict[str, object]", declared).get("content")
    name: object = declared
    if isinstance(name, str):
        pad_id = tokenizer.token_to_id(name)
        if pad_id is not None:
            return pad_id, name

    model_config = read_json(card.path / "config.json", card.model_id)
    pad_id = model_config.get("pad_token_id")
    if isinstance(pad_id, int):
        token = tokenizer.id_to_token(pad_id)
        if isinstance(token, str):
            return pad_id, token

    msg = (
        f"{card.model_id} declares no padding token in tokenizer_config.json or config.json. "
        f"Batches cannot be assembled without one; embed one text at a time, or add "
        f"`pad_token` to the repository's tokenizer configuration."
    )
    raise ConfigError(msg)


def _fetch(model_id: str, revision: str | None) -> tuple[Path, str | None]:
    """The declaration files on local disk, and the commit they came from.

    A local directory is accepted only without a revision claim and records a digest over
    every declaration/tokenizer input this module can read.
    """
    local = Path(model_id).expanduser()
    if local.is_dir():
        if revision is not None:
            raise ConfigError(
                f"embedding.model {model_id!r} is a local directory, so `embedding.revision` "
                "cannot identify it. Remove the revision; local model inputs are identified "
                "by their content digest."
            )
        return local, f"local-sha256:{_local_card_digest(local)}"

    from manicule.embedding.runtimes.hub import snapshot  # noqa: PLC0415 - kept out of import time

    path = snapshot(model_id, CARD_FILES, revision)
    # `snapshot_download` lays files out under `snapshots/<commit>`, so the directory name is
    # the resolved commit — available without a second network call, and available offline.
    resolved = path.name if path.parent.name == "snapshots" else revision
    return path, resolved


def _local_card_digest(path: Path) -> str:
    """Digest every local file that can affect card resolution or tokenization."""
    files = sorted(item for relative in CARD_FILES if (item := path / relative).is_file())
    digest = hashlib.sha256()
    for item in files:
        name = item.relative_to(path).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        file_digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _resolve_pooling(
    model_id: str, declared: dict[str, object], override: Pooling | None
) -> Pooling:
    """The model's declared reduction, or configuration's when the model declares none.

    An override that contradicts the model is refused. A setting that appears to be in force
    and is overruled by a file is bad; a setting that overrules the model's own declaration of
    how it was trained is worse, because it succeeds.
    """
    from_repo = _declared_pooling(model_id, declared)
    if from_repo is not None and override is not None and from_repo is not override:
        msg = (
            f"{model_id} declares {from_repo.value} pooling in 1_Pooling/config.json, but "
            f"configuration asks for {override.value}. The declaration is how the model was "
            f"trained; overriding it produces well-shaped vectors from the wrong reduction. "
            f"Remove the `pooling` setting, or point `embedding.model` at a model that "
            f"declares what you want."
        )
        raise ConfigError(msg)
    if from_repo is not None:
        return from_repo
    if override is not None:
        return override
    msg = (
        f"{model_id} declares no pooling: 1_Pooling/config.json is absent or sets no "
        f"supported flag. Pooling decides whether two sets of vectors are comparable and "
        f"cannot be guessed from a model name — CLS and mean of the same token states differ "
        f"by 0.66-0.80 cosine on a model of this class, with nothing raised. Set "
        f"`pooling` under this embedder's configuration to the reduction the model was "
        f"trained with."
    )
    raise ConfigError(msg)


def _declared_pooling(model_id: str, declared: dict[str, object]) -> Pooling | None:
    unsupported = [flag for flag in _UNSUPPORTED_POOLING_FLAGS if declared.get(flag) is True]
    if unsupported:
        msg = (
            f"{model_id} declares {', '.join(unsupported)}, which manicule does not "
            f"implement. Taking the next flag that happens to be set would index a "
            f"{unsupported[0]} model as something else, so this is refused instead."
        )
        raise ConfigError(msg)

    chosen = [pooling for flag, pooling in _POOLING_FLAGS.items() if declared.get(flag) is True]
    if len(chosen) > 1:
        names = ", ".join(sorted(pooling.value for pooling in chosen))
        msg = (
            f"{model_id} declares more than one pooling mode ({names}). "
            f"Sentence-Transformers concatenates them into one wider vector, which manicule "
            f"does not produce; pick one in configuration."
        )
        raise ConfigError(msg)
    return chosen[0] if chosen else None


def _resolve_dimension(
    model_id: str, config: dict[str, object], pooling_config: dict[str, object]
) -> int:
    """The vector width, cross-checked between the two files that state it."""
    from_pooling = pooling_config.get("word_embedding_dimension")
    from_model = config.get("hidden_size")
    if isinstance(from_pooling, int) and isinstance(from_model, int) and from_pooling != from_model:
        msg = (
            f"{model_id} declares hidden_size {from_model} in config.json and "
            f"word_embedding_dimension {from_pooling} in 1_Pooling/config.json. The vector "
            f"table is created from this number, so it has to be one number."
        )
        raise ConfigError(msg)
    for candidate in (from_pooling, from_model):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    msg = (
        f"{model_id} declares no vector width: neither hidden_size in config.json nor "
        f"word_embedding_dimension in 1_Pooling/config.json is present. manicule never "
        f"assumes a dimension, because a wrong one builds an index that accepts writes."
    )
    raise ConfigError(msg)


def _resolve_length(
    model_id: str,
    path: Path,
    config: dict[str, object],
    special_token_count: int,
    override: int | None,
) -> int:
    """Usable content tokens: declared length, capped by positional capacity, less specials.

    The cap matters in both directions. ``max_position_embeddings`` overstates the limit for
    RoBERTa-family models by the padding offset; and many repositories ship
    ``sentence_bert_config.json`` configured well below what the architecture allows, in which
    case the shipped number is the one that truncates.
    """
    if override is not None:
        return override

    sbert = read_json_if_present(path / "sentence_bert_config.json")
    declared = sbert.get("max_seq_length")
    if not isinstance(declared, int) or declared <= 0:
        msg = (
            f"{model_id} declares no max_seq_length in sentence_bert_config.json. Past a "
            f"model's sequence length input is dropped with no error, so a chunk is indexed "
            f"as its opening tokens while still claiming all of its text. Falling back to "
            f"max_position_embeddings would overstate the limit for most repositories; set "
            f"`max_sequence_length` under this embedder's configuration instead, in usable "
            f"content tokens."
        )
        raise ConfigError(msg)

    capacity = _position_capacity(config)
    attended = min(declared, capacity) if capacity else declared
    usable = attended - special_token_count
    if usable <= 0:
        msg = (
            f"{model_id} has no room for content: a declared length of {declared} against "
            f"{special_token_count} special tokens leaves nothing to embed."
        )
        raise ConfigError(msg)
    return usable


def _position_capacity(config: dict[str, object]) -> int:
    """How many positions the model can actually address, or 0 when it does not say."""
    declared = config.get("max_position_embeddings")
    if not isinstance(declared, int) or declared <= 0:
        return 0
    if str(config.get("model_type")) in _ROBERTA_FAMILY:
        pad = config.get("pad_token_id")
        offset = pad + 1 if isinstance(pad, int) else 2
        return declared - offset
    return declared


def _load_tokenizer(path: Path, model_id: str) -> FastTokenizer:
    # Deferred: `tokenizers` is a Rust extension, and registration must not load it.
    from manicule.embedding.runtimes.tokenization import FastTokenizer  # noqa: PLC0415

    file = path / "tokenizer.json"
    if not file.is_file():
        msg = (
            f"{model_id} ships no tokenizer.json. manicule counts tokens and checks the "
            f"sequence limit with the model's own vocabulary, so a repository without a fast "
            f"tokenizer cannot be used: a budget measured with a stand-in vocabulary "
            f"undercounts, and undercounting is the direction that truncates."
        )
        raise ConfigError(msg)
    return FastTokenizer(file)


def read_json(file: Path, model_id: str) -> dict[str, object]:
    """A declaration file, or a refusal naming what is missing."""
    if not file.is_file():
        msg = (
            f"{model_id} ships no {file.name}, so nothing about it can be read rather than guessed"
        )
        raise ConfigError(msg)
    parsed: object = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"{model_id}'s {file.name} is not a JSON object"
        raise ConfigError(msg)
    return cast("dict[str, object]", parsed)


def read_json_if_present(file: Path) -> dict[str, object]:
    """Absent and unreadable are the same thing here: no declaration, so nothing is claimed."""
    if not file.is_file():
        return {}
    parsed: object = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        return {}
    return cast("dict[str, object]", parsed)


__all__ = [
    "CARD_FILES",
    "ModelCard",
    "load_tokenizer",
    "read_card",
    "read_json",
    "read_json_if_present",
]
