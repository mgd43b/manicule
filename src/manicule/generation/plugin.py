"""The built-in generation plugin.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is, so the extension mechanism is exercised by every installation rather than only by
the people extending it.

**Nothing here imports a provider library.** Registration needs only the configuration model.
An installation that never asks a question never imports litellm, and
``tests/test_import_boundary.py`` fails the build if that stops being true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.generation.config import (
    CLI_GENERATOR_NAME,
    GENERATOR_NAME,
    CliGeneratorConfig,
    GeneratorConfig,
)
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

if TYPE_CHECKING:
    from manicule.core.protocols import Generator


def _build(context: BuildContext) -> Generator:
    # Deferred: this is where the provider library is loaded, and where the token estimator
    # reaches for tiktoken.
    from manicule.config.profiles import profile_config  # noqa: PLC0415
    from manicule.config.providers import ModelRole  # noqa: PLC0415
    from manicule.generation.budget import TokenEstimator  # noqa: PLC0415
    from manicule.generation.prompt import system_message  # noqa: PLC0415
    from manicule.generation.provider import LitellmGenerator  # noqa: PLC0415

    config = context.config
    if not isinstance(config, GeneratorConfig):
        msg = (
            f"a generator was built with {type(config).__name__} where it declares "
            f"{GeneratorConfig.__name__}. Configuration reaching a factory is validated "
            f"against the model the component registered; a factory called outside the "
            f"container has to supply that model itself."
        )
        raise ConfigError(msg)

    settings = context.settings
    llm = settings.llm
    provider = llm.provider.strip().lower()
    endpoint = next(point for point in settings.selected_endpoints if point.role is ModelRole.LLM)
    profile = profile_config(settings.rag.profile, settings.rag.overrides)
    estimator = TokenEstimator(safety_factor=llm.token_safety_factor)
    system_prompt_tokens = estimator.count(system_message(llm.system_prompt_extra)["content"])
    secret = settings.provider(provider).api_key
    return LitellmGenerator(
        llm,
        api_key=secret.get_secret_value() if secret else None,
        base_url=endpoint.base_url,
        egress=endpoint.egress,
        profile=profile,
        profile_name=settings.rag.profile.value,
        system_prompt_tokens=system_prompt_tokens,
        extra_params=dict(config.extra_params),
    )


def _build_cli(context: BuildContext) -> Generator:
    """Build the local command adapter without importing subprocess machinery at discovery."""
    from manicule.config.profiles import profile_config  # noqa: PLC0415
    from manicule.config.providers import ModelRole  # noqa: PLC0415
    from manicule.generation.budget import TokenEstimator  # noqa: PLC0415
    from manicule.generation.cli_provider import CliGenerator  # noqa: PLC0415
    from manicule.generation.prompt import system_message  # noqa: PLC0415

    config = context.config
    if not isinstance(config, CliGeneratorConfig):
        msg = (
            f"a generator was built with {type(config).__name__} where it declares "
            f"{CliGeneratorConfig.__name__}"
        )
        raise ConfigError(msg)

    settings = context.settings
    llm = settings.llm
    endpoint = next(point for point in settings.selected_endpoints if point.role is ModelRole.LLM)
    profile = profile_config(settings.rag.profile, settings.rag.overrides)
    estimator = TokenEstimator(safety_factor=llm.token_safety_factor)
    system_prompt_tokens = estimator.count(system_message(llm.system_prompt_extra)["content"])
    return CliGenerator(
        llm,
        base_url=endpoint.base_url,
        profile=profile,
        profile_name=settings.rag.profile.value,
        system_prompt_tokens=system_prompt_tokens,
    )


class GenerationPlugin:
    """The plugin object the ``generation`` entry point resolves to."""

    manifest = PluginManifest(
        name="generation",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="One generator interface for provider APIs and authenticated local CLIs.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.GENERATOR.named(GENERATOR_NAME),
            _build,
            config_model=GeneratorConfig,
            summary="Streams from Ollama or any hosted provider through one call. Citation "
            "verification sits above it and is not pluggable.",
        )
        registry.add(
            keys.GENERATOR.named(CLI_GENERATOR_NAME),
            _build_cli,
            config_model=CliGeneratorConfig,
            summary="Asks an installed Codex or Claude CLI in non-interactive mode. The same "
            "generator serves the command line, API and browser chat.",
        )


PLUGIN = GenerationPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = ["PLUGIN", "GenerationPlugin"]
