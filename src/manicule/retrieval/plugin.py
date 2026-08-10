"""The built-in retrieval plugin: the two legs, the fusion and the cross-encoder.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party plugin
is, so the extension mechanism is exercised by every installation rather than only by the
people extending it.

**Nothing here imports a model runtime or a tokenizer.** Registration needs the configuration
models and nothing else, so an installation that never runs a query never imports
``sentence-transformers`` or ``tiktoken``. Both wait for the factory that needs them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule.retrieval.config import DenseConfig, FusionConfig, LexicalConfig, RerankConfig
from manicule.retrieval.profile import Profiles

if TYPE_CHECKING:
    from pydantic import BaseModel

    from manicule.core.protocols import Reranker, RetrievalStage

DENSE_NAME = "dense"
LEXICAL_NAME = "lexical"
FUSION_NAME = "rrf"
CROSS_ENCODER_NAME = "cross_encoder"


def _config[T: BaseModel](context: BuildContext, expected: type[T]) -> T:
    """This component's validated configuration, or a refusal naming what arrived instead.

    Substituting defaults would build a stage whose settings appear to be in force and are not,
    which is the failure validation exists to prevent.
    """
    settings = context.config
    if not isinstance(settings, expected):
        msg = (
            f"a retrieval stage was built with {type(settings).__name__} where it declares "
            f"{expected.__name__}. Configuration reaching a factory is validated against the "
            f"model the component registered; a factory called outside the container has to "
            f"supply that model itself."
        )
        raise ConfigError(msg)
    return settings


def _profiles(context: BuildContext) -> Profiles:
    return Profiles(context.settings.rag.overrides)


def build_dense(context: BuildContext) -> RetrievalStage:
    """The dense leg, wired to the embedder and both stores."""
    from manicule.retrieval.dense import DenseStage  # noqa: PLC0415 - keeps registration light

    return DenseStage(
        embedder=context.components.get(keys.EMBEDDER),
        vectors=context.components.get(keys.VECTOR_STORE),
        docstore=context.components.get(keys.DOC_STORE),
        profiles=_profiles(context),
        config=_config(context, DenseConfig),
        name=DENSE_NAME,
    )


def build_lexical(context: BuildContext) -> RetrievalStage:
    """The lexical leg, wired to the authoritative store."""
    from manicule.retrieval.lexical import LexicalStage  # noqa: PLC0415

    _config(context, LexicalConfig)
    return LexicalStage(
        docstore=context.components.get(keys.DOC_STORE),
        profiles=_profiles(context),
        name=LEXICAL_NAME,
    )


def build_fusion(context: BuildContext) -> RetrievalStage:
    """Reciprocal rank fusion, checked against the pipeline it will fuse.

    Raises:
        ConfigError: A named leg is not a stage declared earlier in ``rag.pipeline``. A typo
            would otherwise turn two-leg fusion into one-leg fusion silently — the same failure
            as a leg that returned nothing, with a different cause and the same signature. It
            is checked when the container assembles the pipeline, not on the first query.
    """
    from manicule.retrieval.fusion import RRFStage  # noqa: PLC0415

    config = _config(context, FusionConfig)
    declared = list(context.settings.rag.pipeline)
    position = declared.index(FUSION_NAME) if FUSION_NAME in declared else len(declared)
    earlier = set(declared[:position])
    missing = [leg for leg in config.legs if leg not in earlier]
    if missing:
        msg = (
            f"fusion is configured to fuse {', '.join(missing)}, which rag.pipeline does not "
            f"declare before {FUSION_NAME!r}. Declared before it: "
            f"{', '.join(declared[:position]) or 'nothing'}. A leg that never runs contributes "
            f"no ladder, so fusion would quietly become single-leg and still produce a "
            f"plausible ranking."
        )
        raise ConfigError(msg)
    return RRFStage(config=config, name=FUSION_NAME)


def build_cross_encoder(context: BuildContext) -> Reranker:
    """The cross-encoder reranker. This is where torch is eventually loaded."""
    from manicule.retrieval.rerank import CrossEncoderReranker  # noqa: PLC0415
    from manicule.retrieval.runtimes.cross_encoder import CrossEncoderScorer  # noqa: PLC0415

    config = _config(context, RerankConfig)
    scorer = CrossEncoderScorer(
        config.model,
        batch_size=config.batch_size,
        device=config.device,
        max_length=config.max_length,
    )
    return CrossEncoderReranker(scorer=scorer, profiles=_profiles(context))


class RetrievalPlugin:
    """The plugin object the ``retrieval`` entry point resolves to."""

    manifest = PluginManifest(
        name="retrieval",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="Hybrid retrieval: dense and BM25 legs, reciprocal rank fusion, cross-encoder.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.RETRIEVAL_STAGE.named(DENSE_NAME),
            build_dense,
            config_model=DenseConfig,
            summary="Vector search, scoped by a hydrating join the configuration cannot remove.",
        )
        registry.add(
            keys.RETRIEVAL_STAGE.named(LEXICAL_NAME),
            build_lexical,
            config_model=LexicalConfig,
            summary="BM25 over the authoritative store, filtered before LIMIT.",
        )
        registry.add(
            keys.RETRIEVAL_STAGE.named(FUSION_NAME),
            build_fusion,
            config_model=FusionConfig,
            summary="Reciprocal rank fusion over named legs. Ranks only, never scores.",
        )
        registry.add(
            keys.RERANKER.named(CROSS_ENCODER_NAME),
            build_cross_encoder,
            config_model=RerankConfig,
            summary="sentence-transformers cross-encoder, multilingual by default.",
        )


PLUGIN = RetrievalPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = [
    "CROSS_ENCODER_NAME",
    "DENSE_NAME",
    "FUSION_NAME",
    "LEXICAL_NAME",
    "PLUGIN",
    "RetrievalPlugin",
    "build_cross_encoder",
    "build_dense",
    "build_fusion",
    "build_lexical",
]
