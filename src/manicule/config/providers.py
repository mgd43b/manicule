"""Resolving provider credentials from the environment, by convention.

The convention is ``<PROVIDER>_API_KEY``: provider ``openai`` reads ``OPENAI_API_KEY``.
That rule is applied to *every* provider, including ones manicule has never heard of, so
adding a provider needs no code change here.

:data:`PROVIDER_ALIASES` exists only for providers whose community-standard variable does
not match their name, or that have more than one in circulation. Aliases are tried in
order, and the conventional name is always tried first.

Precedence, highest first:

1. An explicit value in configuration — including ``MANICULE_PROVIDERS__OPENAI__API_KEY``,
   which is configuration expressed as an environment variable.
2. The conventional variable, then any aliases.
3. Nothing. A missing key is not an error here: whether it matters depends on whether that
   provider is selected, which :func:`manicule.config.settings.Settings.policy_problems`
   decides once the whole configuration is known.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr

PROVIDER_ALIASES: Mapping[str, tuple[str, ...]] = {
    "google": ("GEMINI_API_KEY",),
    "xai": ("GROK_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "together": ("TOGETHERAI_API_KEY",),
}
"""Extra variables to try, after the conventional ``<PROVIDER>_API_KEY``."""

LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "mlx", "onnx", "local"})
"""Providers that run on this machine and need no credential.

Selecting one of these is also what satisfies a local-only data policy: nothing leaves.
"""

DEFAULT_BASE_URLS: Mapping[str, str] = {
    "ollama": "http://localhost:11434",
}
"""Base URLs manicule fills in.

Only for endpoints that are conventionally fixed and locally hosted. Hosted providers are
reached through the generation library's own defaults, so that a URL manicule pins today
cannot become a URL that is wrong tomorrow.
"""


def env_var_names(provider: str) -> tuple[str, ...]:
    """Environment variables consulted for ``provider``'s key, in order."""
    conventional = provider.strip().upper().replace("-", "_").replace(".", "_") + "_API_KEY"
    return (conventional, *PROVIDER_ALIASES.get(provider.strip().lower(), ()))


def is_local(provider: str) -> bool:
    """Whether ``provider`` runs on this machine, and therefore needs no credential."""
    return provider.strip().lower() in LOCAL_PROVIDERS


class ProviderSettings(BaseModel):
    """Credentials and endpoint for one model provider.

    One shape for every provider. There is no per-vendor settings type, because a per-vendor
    type in configuration becomes a per-vendor branch in every consumer.
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = Field(
        default=None,
        description="Resolved from the environment when not set here. Held as a secret so "
        "that printing configuration cannot leak it.",
    )
    base_url: str | None = Field(
        default=None,
        description="Endpoint override. Local and hosted models differ by this and nothing else.",
    )
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Additional provider parameters passed through untouched, e.g. an "
        "organisation id or an API version.",
    )

    @property
    def has_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())


def resolve_provider_keys(
    providers: Mapping[str, ProviderSettings],
    required: frozenset[str] = frozenset(),
    environ: Mapping[str, str] | None = None,
) -> dict[str, ProviderSettings]:
    """Fill in missing credentials and base URLs from the environment.

    Args:
        providers: Provider settings as configuration gave them.
        required: Providers that configuration selects. These get an entry even if
            configuration never mentioned them, so that ``LLM_PROVIDER=openai`` plus
            ``OPENAI_API_KEY`` in the environment is a complete setup with no config file.
        environ: Environment to read. Defaults to the real one.

    Returns:
        A new mapping. Entries that already had a key keep it — an explicit value is never
        overwritten by a conventional one.
    """
    env = os.environ if environ is None else environ
    resolved: dict[str, ProviderSettings] = {}

    for name in sorted(set(providers) | {p.strip().lower() for p in required if p.strip()}):
        settings = providers.get(name, ProviderSettings())
        api_key = settings.api_key
        if api_key is None and not is_local(name):
            for var in env_var_names(name):
                value = env.get(var)
                if value:
                    api_key = SecretStr(value)
                    break
        base_url = settings.base_url or DEFAULT_BASE_URLS.get(name)
        resolved[name] = settings.model_copy(update={"api_key": api_key, "base_url": base_url})

    return resolved


__all__ = [
    "DEFAULT_BASE_URLS",
    "LOCAL_PROVIDERS",
    "PROVIDER_ALIASES",
    "ProviderSettings",
    "env_var_names",
    "is_local",
    "resolve_provider_keys",
]
