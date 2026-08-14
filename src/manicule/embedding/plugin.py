"""The built-in embedding plugin: the MLX and onnxruntime backends.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is, so the extension mechanism is exercised by every installation.

**Nothing here imports a model runtime.** Registration needs only the configuration model, so
an installation that never selects an embedder never imports MLX, onnxruntime or numpy. The
runtime is imported inside the backend's own ``setup``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.embedding.config import EmbedderConfig, MlxEmbedderConfig
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

if TYPE_CHECKING:
    from manicule.core.protocols import Embedder
    from manicule.embedding.cards import ModelCard

MLX_NAME = "mlx"
ONNX_NAME = "onnx"


def _card(context: BuildContext) -> tuple[ModelCard, EmbedderConfig]:
    """Read the model's declaration at construction time.

    Deliberately at construction rather than at first use. The chunker takes the embedder as a
    construction dependency and refuses to start when its token budget exceeds this model's
    sequence limit — a check that has to happen before a corpus is built differently on this
    machine than on another. It costs a few kilobytes of metadata; the weights wait for setup.

    Raises:
        ConfigError: The context carries configuration of some other type. Substituting
            defaults here would build an embedder whose settings appear to be in force and are
            not, which is what validation exists to prevent.
    """
    # Deferred: cards.py reaches for `tokenizers` and huggingface-hub, and registration must
    # stay free of both.
    from manicule.embedding.cards import read_card  # noqa: PLC0415

    settings = context.config
    if not isinstance(settings, EmbedderConfig):
        msg = (
            f"an embedder was built with {type(settings).__name__} where it declares "
            f"{EmbedderConfig.__name__}. Configuration reaching a factory is validated against "
            f"the model the component registered; a factory called outside the container has "
            f"to supply that model itself."
        )
        raise ConfigError(msg)

    embedding = context.settings.embedding
    card = read_card(
        embedding.model,
        revision=embedding.revision,
        pooling_override=settings.pooling,
        max_sequence_length_override=settings.max_sequence_length,
    )
    return card, settings


def _build_mlx(context: BuildContext) -> Embedder:
    # Deferred: this is where MLX and Metal are loaded.
    from manicule.embedding.runtimes.mlx_backend import MEGABYTE, MlxEmbedder  # noqa: PLC0415

    if not isinstance(context.config, MlxEmbedderConfig):
        # Checked before anything is read, because the registry validates against the model a
        # component registered and so this is the factory being called from outside the
        # container. Falling back to the shared model's defaults would silently drop the cache
        # bound, which is the one setting here that stops a run from taking the machine down.
        msg = (
            f"the mlx embedder was built with {type(context.config).__name__} where it "
            f"declares {MlxEmbedderConfig.__name__}. Its cache bound would not be applied."
        )
        raise ConfigError(msg)
    config = context.config
    card, _ = _card(context)
    embedding = context.settings.embedding
    return MlxEmbedder(
        card,
        weights=config.weights,
        batch_size=embedding.batch_size,
        cache_entries=embedding.cache_entries,
        cache_limit_bytes=config.cache_limit_mb * MEGABYTE,
    )


def _build_onnx(context: BuildContext) -> Embedder:
    # Deferred: this is where onnxruntime is loaded.
    from manicule.embedding.runtimes.onnx_backend import OnnxEmbedder  # noqa: PLC0415

    card, config = _card(context)
    embedding = context.settings.embedding
    return OnnxEmbedder(
        card,
        weights=config.weights,
        batch_size=embedding.batch_size,
        cache_entries=embedding.cache_entries,
    )


class EmbeddingPlugin:
    """The plugin object the ``embedding`` entry point resolves to."""

    manifest = PluginManifest(
        name="embedding",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="MLX and onnxruntime embedders, pooling in manicule's own numpy.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.EMBEDDER.named(MLX_NAME),
            _build_mlx,
            config_model=MlxEmbedderConfig,
            summary="Metal-native, in-process, on Apple Silicon. Pools from token states.",
        )
        registry.add(
            keys.EMBEDDER.named(ONNX_NAME),
            _build_onnx,
            config_model=EmbedderConfig,
            summary="Portable, in-process, and the reference the MLX backend is measured "
            "against. Pools from token states.",
        )


PLUGIN = EmbeddingPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = ["MLX_NAME", "ONNX_NAME", "PLUGIN", "EmbeddingPlugin"]
