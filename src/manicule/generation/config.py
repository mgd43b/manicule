"""The generator's own configuration model.

Separate from :mod:`manicule.generation.provider` for the reason every other component here
does the same: registration needs the model so that settings written for this component are
*validated* rather than ignored, and registration runs in every process that starts. This
module imports nothing heavier than pydantic, so plugin discovery never loads a provider
library.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

GENERATOR_NAME = "litellm"
CLI_GENERATOR_NAME = "cli"


class GeneratorConfig(BaseModel):
    """Per-component settings for the built-in generator.

    Almost everything lives on ``llm`` in the main configuration tree, because it applies to
    whichever generator is bound. What is here is the one thing that cannot: parameters
    specific to a provider's own API.
    """

    model_config = ConfigDict(extra="forbid")

    extra_params: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional parameters passed to the provider call untouched — an API "
        "version, an organization id, a provider-specific sampling knob. Deliberately opaque: "
        "the alternative is a per-vendor settings type, which becomes a per-vendor branch in "
        "every consumer. Anything manicule itself sets (model, messages, stream, the three "
        "timeouts, num_ctx) is set after these, so this cannot quietly override a guarantee.",
    )


class CliGeneratorConfig(BaseModel):
    """Configuration for the built-in Codex and Claude command-line adapter.

    The command is intentionally selected by ``llm.provider`` rather than accepted as an
    arbitrary argv here.  Besides keeping configuration small, that lets the adapter put the
    two supported CLIs into their non-interactive, non-writing modes without an extra argument
    quietly undoing those guarantees.
    """

    model_config = ConfigDict(extra="forbid")


__all__ = ["CLI_GENERATOR_NAME", "GENERATOR_NAME", "CliGeneratorConfig", "GeneratorConfig"]
