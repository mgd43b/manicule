"""The built-in embedding plugin: the onnxruntime backend.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is, so the extension mechanism is exercised by every installation.

**The MLX backend is no longer here.** It links ``mlx-embeddings``, which is GPL-3.0, so it
ships as its own GPL-3.0-or-later distribution — ``manicule-mlx`` — and claims the
``embedder.mlx`` slot through this same entry-point group. manicule has no special knowledge
that it exists; ``[embedding] provider = "mlx"`` resolves through discovery like any other
component. :func:`read_embedder_card` and :func:`embedder_metadata_factory` are public because
that package calls them: they are the shared half of building an embedder, and duplicating
them there would let the two drift on how a model's declaration is read.

**Nothing here imports a model runtime.** Registration needs only the configuration model, so
an installation that never selects an embedder never imports onnxruntime or numpy. The runtime
is imported inside the backend's own ``setup``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.embedding.config import EmbedderConfig
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

if TYPE_CHECKING:
    from manicule.core.embedding import EmbedFingerprint
    from manicule.core.protocols import Embedder
    from manicule.embedding.cards import ModelCard
    from manicule.plugins.registry import MetadataContext, MetadataFactory

ONNX_NAME = "onnx"


def read_embedder_card(context: BuildContext) -> tuple[ModelCard, EmbedderConfig]:
    """Read the model's declaration at construction time.

    Public because an out-of-tree backend needs it — ``manicule-mlx`` is the first — and a
    second copy of this logic would be a second answer to "what does this model declare".

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
    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415
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
    revision = embedding.revision or builtin_model_revision(embedding.model)
    card = read_card(
        embedding.model,
        revision=revision,
        pooling_override=settings.pooling,
        max_sequence_length_override=settings.max_sequence_length,
    )
    return card, settings


def _build_onnx(context: BuildContext) -> Embedder:
    # Deferred: this is where onnxruntime is loaded.
    from manicule.embedding.runtimes.onnx_backend import OnnxEmbedder  # noqa: PLC0415

    card, config = read_embedder_card(context)
    embedding = context.settings.embedding
    return OnnxEmbedder(
        card,
        weights=config.weights,
        weights_revision=config.weights_revision,
        batch_size=embedding.batch_size,
        cache_entries=embedding.cache_entries,
    )


def embedder_metadata_factory(provider: str) -> MetadataFactory:
    """Declare the exact configured vector space from local metadata only.

    Parameterized by ``provider`` and public for the same reason as :func:`read_embedder_card`:
    an out-of-tree backend has to produce a fingerprint the same way this one does, or two
    backends would describe the same vector space differently and an index would refuse a
    runtime that agrees with it.
    """

    def metadata(context: MetadataContext) -> EmbedFingerprint:
        from manicule.embedding.artifacts import (  # noqa: PLC0415
            builtin_model_revision,
            describe_artifact,
        )
        from manicule.embedding.cards import read_cached_card  # noqa: PLC0415

        settings = context.config
        if not isinstance(settings, EmbedderConfig):
            raise ConfigError(
                f"embedder metadata expected {EmbedderConfig.__name__}, got "
                f"{type(settings).__name__}"
            )
        embedding = context.settings.embedding
        revision = embedding.revision or builtin_model_revision(embedding.model)
        try:
            card = read_cached_card(
                embedding.model,
                revision=revision,
                pooling_override=settings.pooling,
                max_sequence_length_override=settings.max_sequence_length,
            )
            artifact = describe_artifact(
                provider,
                card.source_ref,
                card.revision,
                override=settings.weights,
                revision=settings.weights_revision,
            )
        except ConfigError as exc:
            raise ConfigError(
                "configured embedding identity is unavailable for metadata-only rebuild planning"
            ) from exc
        return card.fingerprint(
            backend=provider,
            weights_ref=artifact.ref,
            weights_identity=artifact.identity,
        )

    return metadata


class EmbeddingPlugin:
    """The plugin object the ``embedding`` entry point resolves to."""

    manifest = PluginManifest(
        name="embedding",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="The onnxruntime embedder, pooling in manicule's own numpy.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.EMBEDDER.named(ONNX_NAME),
            _build_onnx,
            config_model=EmbedderConfig,
            metadata_factory=embedder_metadata_factory(ONNX_NAME),
            summary="Portable, in-process, and the reference every other backend is measured "
            "against. Pools from token states.",
        )


PLUGIN = EmbeddingPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = [
    "ONNX_NAME",
    "PLUGIN",
    "EmbeddingPlugin",
    "embedder_metadata_factory",
    "read_embedder_card",
]
