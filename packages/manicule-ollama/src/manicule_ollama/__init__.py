"""The Ollama embedding backend, registered as an ordinary third-party manicule plugin.

There is no shorter route into manicule than this one. The backend claims the
``embedder.ollama`` slot through the public ``manicule.plugins`` entry-point group, exactly as
any other plugin would, and ``[embedding] provider = "ollama"`` resolves to it through ordinary
discovery. Nothing under ``src/manicule`` knows this package exists — not even
``manicule.plugins.registry.KNOWN_DISTRIBUTIONS``, which carries an install hint for ``mlx``
only because ``mlx`` was manicule's default before that backend moved out and so is a
configuration people already have. Naming a package there that PyPI does not serve would be a
worse error than the one it fixes.

**Why this is a separate distribution at all**, since unlike ``manicule-mlx`` it carries no
copyleft dependency and could have lived in-tree: it is the only embedder whose model is a
*remote service*. It brings an HTTP client, a deployment topology, and a set of failures — an
unreachable host, a model somebody re-pulled — that an installation embedding in-process should
neither resolve nor reason about.

**What it is for.** Embedding moves off the machine running manicule when that machine is the
wrong place for it: the case this was written for is a pod on Ivy Bridge Xeons — no AVX2, which
is what onnxruntime's fast kernels are built around — next to a GPU node that already runs
Ollama for generation.

**Nothing here imports httpx.** Registration needs only the configuration model, so an
installation that has this package present but selects ``onnx`` never opens a socket or
constructs a client. The runtime arrives inside the factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule_ollama.config import OllamaEmbedderConfig

if TYPE_CHECKING:
    from manicule.core.embedding import EmbedFingerprint
    from manicule.core.protocols import Embedder
    from manicule.plugins.registry import MetadataContext

OLLAMA_NAME = "ollama"


def build_ollama(context: BuildContext) -> Embedder:
    """Construct the backend, reading the server once so the fingerprint can exist.

    The read happens here rather than in ``setup`` because the chunker takes the embedder as a
    construction dependency and refuses to start when its token budget exceeds this model's
    sequence limit — a refusal that has to happen before a corpus is built differently on this
    machine than on another. For the built-in backends that costs a few kilobytes of model
    declaration; here it costs three HTTP round trips, one of which may prompt the server to
    load the model. No weights reach this process either way.

    The declaration is written to the cache directory on the way through, because metadata-only
    rebuild planning has to derive the same identity later without a network — and a served
    model has no equivalent of the Hugging Face cache to read it from.
    """
    # Deferred: importing these is what pulls in httpx and the tokenizer.
    from manicule_ollama.backend import OllamaEmbedder  # noqa: PLC0415
    from manicule_ollama.client import OllamaClient  # noqa: PLC0415
    from manicule_ollama.served import record, resolve  # noqa: PLC0415

    config = context.config
    if not isinstance(config, OllamaEmbedderConfig):
        # Checked before anything is read, because the registry validates against the model a
        # component registered and so this is the factory being called from outside the
        # container. Falling back to the shared model's defaults would silently drop
        # `base_url`, `tokenizer` and `num_ctx` — which is every setting that decides where the
        # vectors come from and what they mean.
        msg = (
            f"the ollama embedder was built with {type(config).__name__} where it declares "
            f"{OllamaEmbedderConfig.__name__}. Its server address, tokenizer and served "
            f"context would not be applied."
        )
        raise ConfigError(msg)

    embedding = context.settings.embedding
    client = OllamaClient(
        config.base_url,
        timeout_s=config.timeout_s,
        connect_timeout_s=config.connect_timeout_s,
    )
    served = resolve(client, embedding.model, config)
    record(served, client, context.cache_dir)
    return OllamaEmbedder(
        served,
        client,
        cache_dir=context.cache_dir,
        keep_alive=config.keep_alive,
        batch_size=embedding.batch_size,
        cache_entries=embedding.cache_entries,
    )


def ollama_metadata(context: MetadataContext) -> EmbedFingerprint:
    """Declare the configured vector space from the recorded declaration, with no network.

    The built-in backends answer this from a model card already in the Hugging Face cache
    (:func:`manicule.embedding.plugin.embedder_metadata_factory`). A served model has no cache
    of its own, so :func:`manicule_ollama.served.record` writes one every time the server is
    read and this reads it back. Absent, it refuses and says which command will produce it —
    rather than contacting the server, which is the thing a metadata-only path exists to avoid.
    """
    from manicule_ollama.served import cached_fingerprint  # noqa: PLC0415

    config = context.config
    if not isinstance(config, OllamaEmbedderConfig):
        raise ConfigError(
            f"ollama embedder metadata expected {OllamaEmbedderConfig.__name__}, got "
            f"{type(config).__name__}"
        )
    return cached_fingerprint(
        context.cache_dir, config.base_url, context.settings.embedding.model, config
    )


class OllamaPlugin:
    """The plugin object the ``ollama`` entry point resolves to."""

    manifest = PluginManifest(
        name=OLLAMA_NAME,
        version="0.1.0",
        # The range of manicule whose contracts this backend is written against. Narrower in
        # effect than the name suggests: it implements the `Embedder` protocol, but it also
        # supplies `card` and `count_tokens`, which `manicule.ingest.workers.worker_config` and
        # `manicule.chunking.tokens.SupportsTokenCount` read structurally. Those are the shapes
        # that would break here as a silently provisional chunker rather than as an import
        # error.
        core_version=">=0.1,<0.2",
        summary="Embedding on an Ollama server, for hosts that cannot embed in process.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.EMBEDDER.named(OLLAMA_NAME),
            build_ollama,
            config_model=OllamaEmbedderConfig,
            metadata_factory=ollama_metadata,
            summary="Served by an Ollama host. The server pools, so this is a tier B backend "
            "and its claims are measured against the server at setup.",
        )


PLUGIN = OllamaPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = [
    "OLLAMA_NAME",
    "PLUGIN",
    "OllamaEmbedderConfig",
    "OllamaPlugin",
    "build_ollama",
    "ollama_metadata",
]
