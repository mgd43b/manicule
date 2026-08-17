"""The MLX embedding backend, registered as an ordinary third-party manicule plugin.

There is no shorter route into manicule than this one. The backend claims the ``embedder.mlx``
slot through the public ``manicule.plugins`` entry-point group, exactly as any other plugin
would, and ``[embedding] provider = "mlx"`` resolves to it through ordinary discovery.

manicule knows this package's *name* in exactly one place, and it is not a structural
dependency: ``manicule.plugins.registry.KNOWN_DISTRIBUTIONS`` maps ``embedder.mlx`` to an
install hint, so that a configuration naming ``mlx`` on an installation without it gets "install
manicule-mlx" rather than a correct but unhelpful "no embedder named 'mlx'. Available: onnx".
`mlx` was manicule's default until this package existed, so that configuration is a thing people
actually have. Nothing else in manicule refers to it, and `tests/test_license_boundary.py`
fails if manicule ever grows a real dependency on it.

**Nothing here imports MLX.** Registration needs only the configuration model, so an
installation that has this package present but selects ``onnx`` never loads Metal. The runtime
arrives inside the backend's own ``setup``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.embedding.plugin import embedder_metadata_factory, read_embedder_card
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule_mlx.config import MlxEmbedderConfig

if TYPE_CHECKING:
    from manicule.core.protocols import Embedder

MLX_NAME = "mlx"


def build_mlx(context: BuildContext) -> Embedder:
    """Construct the backend. The one place MLX and Metal are loaded."""
    # Deferred: importing this module is what pulls in mlx-embeddings.
    from manicule_mlx.backend import MEGABYTE, MlxEmbedder  # noqa: PLC0415

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
    card, _ = read_embedder_card(context)
    embedding = context.settings.embedding
    return MlxEmbedder(
        card,
        weights=config.weights,
        weights_revision=config.weights_revision,
        batch_size=embedding.batch_size,
        cache_entries=embedding.cache_entries,
        cache_limit_bytes=config.cache_limit_mb * MEGABYTE,
    )


class MlxPlugin:
    """The plugin object the ``mlx`` entry point resolves to."""

    manifest = PluginManifest(
        name=MLX_NAME,
        version="0.1.0",
        # The range of manicule whose `PooledEmbedder` contract this backend is written
        # against. It matters more here than for a plugin that only touches public protocols:
        # this one subclasses a base class and implements its underscore-prefixed methods, so a
        # change to that shape lands here as a broken backend rather than a missing attribute.
        core_version=">=0.1,<0.2",
        summary="Metal-native embedding on Apple Silicon. GPL-3.0-or-later.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.EMBEDDER.named(MLX_NAME),
            build_mlx,
            config_model=MlxEmbedderConfig,
            metadata_factory=embedder_metadata_factory(MLX_NAME),
            summary="Metal-native, in-process, on Apple Silicon. Pools from token states.",
        )


PLUGIN = MlxPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = ["MLX_NAME", "PLUGIN", "MlxEmbedderConfig", "MlxPlugin", "build_mlx"]
