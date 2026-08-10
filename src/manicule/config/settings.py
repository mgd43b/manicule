"""The configuration tree.

One declarative layer: the same models validate the config file, the environment, plugin
configuration and API payloads. There is no second schema to keep in step.

**Sources layer rather than compete.** Highest priority first: values passed in code, then
environment variables, then ``.env`` files, then the config file, then defaults. Setting one
field in the environment overrides that field and leaves the rest of the file in force —
a file and an environment are two halves of one configuration, not two rival ones.

Environment variables use the ``MANICULE_`` prefix and ``__`` for nesting, so
``MANICULE_SECURITY__AUTH__MODE=api_key`` sets ``security.auth.mode``. Provider credentials
additionally follow the conventional ``<PROVIDER>_API_KEY`` names; see
:mod:`manicule.config.providers`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, override

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from manicule.config.providers import (
    ProviderSettings,
    env_var_names,
    is_local,
    resolve_provider_keys,
)
from manicule.core.errors import PolicyError
from manicule.core.retrieval import RetrievalProfile

ENV_PREFIX = "MANICULE_"
APP_NAME = "manicule"


def _xdg(var: str, default: str) -> Path:
    raw = os.environ.get(var)
    root = Path(raw).expanduser() if raw else Path.home() / default
    return root / APP_NAME


def default_config_dir() -> Path:
    """Where the config file and a config-scoped ``.env`` live."""
    override = os.environ.get(f"{ENV_PREFIX}CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_CONFIG_HOME", ".config")


def default_data_dir() -> Path:
    """Where the database, vector index and retained source bytes live."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def default_cache_dir() -> Path:
    """Where regenerable artefacts live. Safe to delete."""
    return _xdg("XDG_CACHE_HOME", ".cache")


def config_file() -> Path:
    """The config file path.

    ``MANICULE_CONFIG_FILE`` wins; then ``manicule.toml`` beside the working directory, so a
    project can carry its own; then the user's config directory.
    """
    override = os.environ.get(f"{ENV_PREFIX}CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    local = Path.cwd() / f"{APP_NAME}.toml"
    if local.is_file():
        return local
    return default_config_dir() / "config.toml"


class PrefixedDotEnvSource(DotEnvSettingsSource):
    """A ``.env`` source that reads manicule's variables and ignores everybody else's.

    A ``.env`` file is shared ground: it holds ``OPENAI_API_KEY``, database URLs and
    whatever else the project needs. Treating every unrecognised line as a misspelled
    manicule setting would make a normal ``.env`` file unloadable, so only ``MANICULE_``
    names are considered here. Unknown *prefixed* names are still rejected, which is where
    a typo actually shows up.
    """

    @override
    def _load_env_vars(self) -> Mapping[str, str | None]:
        prefix = self.env_prefix.lower()
        return {
            name: value
            for name, value in super()._load_env_vars().items()
            if name.lower().startswith(prefix)
        }


def env_files() -> tuple[Path, ...]:
    """``.env`` files to read, lowest priority first.

    The user's config directory holds the credentials that apply everywhere; the working
    directory holds the ones that apply to this project, and wins. The real environment wins
    over both, so an exported variable always beats a file.
    """
    return (default_config_dir() / ".env", Path.cwd() / ".env")


def provider_environment() -> Mapping[str, str]:
    """Environment used for credential resolution: ``.env`` files, then the real environment.

    Provider keys follow their own conventional names (``OPENAI_API_KEY``), not manicule's
    prefixed ones, so they are read here rather than by the settings sources — which only see
    variables beginning with ``MANICULE_``.
    """
    merged: dict[str, str] = {}
    for path in env_files():
        for key, value in dotenv_values(path, encoding="utf-8").items():
            if value is not None:
                merged[key] = value
    merged.update(os.environ)
    return merged


# --- leaf sections -------------------------------------------------------------------------


class Section(BaseModel):
    """Base for configuration sections: unknown keys are rejected, not ignored.

    A typo in a config file that silently does nothing is worse than one that fails at
    startup, because the setting appears to be in force.
    """

    model_config = ConfigDict(extra="forbid")


class Mode(StrEnum):
    """Whether this installation serves one person or a team."""

    PERSONAL = "personal"
    TEAM = "team"


class Theme(StrEnum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class UiSettings(Section):
    theme: Theme = Theme.AUTO
    locale: str = Field(default="auto", description="``auto`` follows the client's request.")


class TelemetrySettings(Section):
    enabled: bool = Field(default=False, description="Off unless switched on, deliberately.")
    endpoint: str | None = None


class AuditDestination(StrEnum):
    LOCAL = "local"
    SYSLOG = "syslog"
    WEBHOOK = "webhook"


class AuditSettings(Section):
    enabled: bool = False
    events: tuple[str, ...] = Field(
        default=(), description="Event names to record. Empty means all of them."
    )
    destination: AuditDestination = AuditDestination.LOCAL


class RedactionMethod(StrEnum):
    REPLACE = "replace"
    HASH = "hash"
    REMOVE = "remove"


class RedactionSettings(Section):
    """Personal-data redaction.

    Applied where the outbound context is assembled, so it governs what is *sent to a model*
    and leaves the index intact. Redacting at ingest would destroy the stored document
    permanently while doing nothing about what the model later sees, which is the opposite of
    what the setting's name promises.
    """

    enabled: bool = False
    patterns: tuple[str, ...] = Field(
        default=(),
        description="Named detectors, e.g. ``email``, ``phone``, ``credit-card``.",
    )
    method: RedactionMethod = RedactionMethod.REPLACE
    replacement: str = "[REDACTED]"


class SourceRestrictions(Section):
    """Per-source overrides on where a document's content may be processed."""

    local_only: tuple[str, ...] = Field(
        default=(), description="Sources whose content must never reach a hosted model."
    )
    cloud_allowed: tuple[str, ...] = Field(
        default=(), description="Sources exempted from a local-only default."
    )


class WorkspaceOverride(Section):
    """Per-workspace policy overrides."""

    cloud_allowed: bool | None = None


class DataPolicySettings(Section):
    cloud_allowed: bool = Field(
        default=True,
        description="Whether document content may be sent to a hosted model at all.",
    )
    auto_redact: RedactionSettings = Field(default_factory=RedactionSettings)
    source_restrictions: SourceRestrictions = Field(default_factory=SourceRestrictions)
    workspace_overrides: dict[str, WorkspaceOverride] = Field(default_factory=dict)


class AuthMode(StrEnum):
    NONE = "none"
    API_KEY = "api_key"
    OAUTH = "oauth"


class Role(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class OAuthProvider(Section):
    type: Literal["google", "github"]
    client_id: str = Field(min_length=1)
    client_secret: SecretStr
    redirect_uri: str | None = None
    workspace: str | None = None
    role: Role = Role.MEMBER
    allowed_emails: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    allow_any_user: bool = False


class AuthSettings(Section):
    mode: AuthMode = Field(
        default=AuthMode.NONE,
        description="``none`` is only permitted while every interface is bound to loopback.",
    )
    providers: tuple[OAuthProvider, ...] = ()
    session_max_age_s: int = Field(default=60 * 60 * 24 * 7, ge=60)


class TransportSettings(Section):
    """How manicule is reachable over the network."""

    bind_host: str = Field(
        default="127.0.0.1",
        description="Loopback by default. An unauthenticated service on a routable address "
        "is an open document index, so binding wider requires authentication to be on.",
    )
    port: int = Field(default=8765, ge=1, le=65535)
    enforce_https: bool = True
    trusted_proxies: tuple[str, ...] = Field(
        default=(),
        description="CIDR ranges whose forwarded-for headers are believed. Empty means none "
        "are — an unrestricted trust in that header is a trivial identity spoof.",
    )
    allowed_origins: tuple[str, ...] = ()
    allowed_endpoints: tuple[str, ...] = Field(
        default=(),
        description="Outbound hosts plugins and connectors may reach. Empty is unrestricted.",
    )
    widget_allowed_domains: tuple[str, ...] = ()

    @property
    def is_loopback(self) -> bool:
        return self.bind_host in {"127.0.0.1", "::1", "localhost"}


class AtRestSettings(Section):
    redact_logs_content: bool = Field(default=True, description="Keep document text out of logs.")


class SecuritySettings(Section):
    auth: AuthSettings = Field(default_factory=AuthSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    data_policy: DataPolicySettings = Field(default_factory=DataPolicySettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    storage: AtRestSettings = Field(default_factory=AtRestSettings)


class WebhookSettings(Section):
    url: str = Field(min_length=1)
    events: tuple[str, ...] = Field(min_length=1)
    secret: SecretStr | None = None
    retries: int = Field(default=2, ge=0, le=10)


class EventSettings(Section):
    transport: Literal["in_process", "webhook"] = "in_process"
    webhooks: tuple[WebhookSettings, ...] = ()


class StorageSettings(Section):
    """Where indexed data lives.

    The choices are closed: a relational store for the sixteen tables of collections, tags,
    versions and audit records, and a vector store for vectors. Naming the alternatives in
    configuration would advertise support that does not exist.
    """

    db: Literal["sqlite"] = "sqlite"
    db_url: str | None = Field(
        default=None, description="Overrides the default path under the data directory."
    )
    vector_db: Literal["lancedb"] = "lancedb"
    vector_db_url: str | None = None
    retain_source_bytes: bool = Field(
        default=True,
        description="Keep fetched bytes so re-parsing never means re-fetching. Turning this "
        "off makes every re-index a re-crawl.",
    )


class IngestSettings(Section):
    """How the pipeline runs: limits, concurrency, and the cadence of the sweeps.

    Every default here is a number the design argued for rather than a round one that felt
    safe. Where a value bounds a resource, the tunable is the quantity that maps to the
    resource — ``target_batch_tokens`` rather than a batch count — because a batch size is a
    proxy for memory and a bad one: thirty-two chunks of 512 tokens and thirty-two of 8 000
    are very different allocations.
    """

    fetch_concurrency: int = Field(
        default=8, ge=1, description="In-flight fetches per connector. Bounded by the remote."
    )
    parse_workers: int = Field(
        default=0,
        ge=0,
        description="Parse worker subprocesses. ``0`` derives ``min(4, cpu_count - 1)``, "
        "never fewer than one.",
    )
    parse_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="Wall clock for one parser attempt, not for the document. A chain of "
        "three parsers may legitimately take three times this before the document fails; a "
        "per-document limit would make the last parser fail for the first parser's reasons.",
    )
    parse_memory_limit_mb: int = Field(
        default=1024,
        ge=64,
        description="Resident memory one parse worker may reach before it is killed.",
    )
    memory_poll_interval_s: float = Field(
        default=0.25,
        gt=0,
        description="How often the parent samples a worker's memory where the kernel will "
        "not enforce a limit for it. Sampling can overshoot between ticks, which is accepted: "
        "the goal is to stop a runaway before it takes the machine down, not a byte-exact quota.",
    )
    max_documents_per_worker: int = Field(
        default=500,
        ge=1,
        description="Recycle a worker after this many documents, to bound leaks in native "
        "parser libraries — a category of bug no amount of care in manicule prevents.",
    )
    max_fetch_bytes: int = Field(
        default=256 * 1024 * 1024, ge=1, description="Refuse a fetched body larger than this."
    )
    target_batch_tokens: int = Field(
        default=16_384,
        ge=1,
        description="Tokens per embedding batch. The batch *size* is derived from this and "
        "the chunk budget, because the quantity that maps to memory is tokens, not chunks.",
    )
    max_embed_batch: int = Field(default=64, ge=1, description="Upper clamp on the derived size.")
    queue_depth_factor: int = Field(
        default=2,
        ge=1,
        description="Bounded queue depth, as a multiple of the consumer's parallelism. "
        "Bounded so that backpressure reaches discovery: an unbounded queue turns a slow "
        "embedder into unbounded memory growth *and* lets a connector race ahead until its "
        "pagination cursors expire.",
    )
    stale_after_s: float = Field(
        default=3600.0,
        gt=0,
        description="How long a document may sit in an in-flight status before the recovery "
        "sweep requeues it. Comfortably above any per-document limit.",
    )
    shutdown_grace_s: float = Field(
        default=30.0,
        ge=0,
        description="How long in-flight documents get to finish on cancellation. A document "
        "mid-embed is close to done and finishing it is cheaper than redoing it.",
    )
    reconcile_interval_s: float = Field(
        default=7 * 24 * 3600.0,
        gt=0,
        description="Deletion detection that runs only when someone remembers is deletion "
        "detection that does not run.",
    )
    reconcile_max_delete_fraction: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Refuse a reconciliation proposing to delete more than this share of a "
        "connector's live documents, and record the proposal for confirmation.",
    )
    soft_delete_grace_s: float = Field(
        default=30 * 24 * 3600.0,
        ge=0,
        description="How long a soft-deleted document's chunks survive before the sweep "
        "purges them. Free restore inside it; a re-parse from retained bytes outside it. "
        "Unbounded free restore would mean unbounded dilution of every vector search.",
    )
    sweep_interval_s: float = Field(
        default=3600.0,
        gt=0,
        description="How often the vector sweep runs. Scheduled rather than triggered by "
        "deletion, so a large reconciliation does not produce a sweep storm during a sync.",
    )
    sweep_batch: int = Field(default=1000, ge=1, description="Tombstones retired per sweep pass.")
    watch_debounce_s: float = Field(
        default=0.5,
        gt=0,
        description="Coalescing window for filesystem events. Editors do not write files the "
        "way the naive model assumes: one logical save commonly produces several events, and "
        "ingesting on the first indexes a partial or empty document.",
    )


class EmbeddingSettings(Section):
    """The embedding runtime.

    **There is no dimension setting, and there will not be one.** The dimension is a property
    of the model, read from the embedder's fingerprint at run time. A configurable dimension
    is a value that can disagree with the model, and when it does the index is silently
    wrong.
    """

    provider: str = Field(default="mlx", description="Which embedder implementation to use.")
    model: str = Field(default="BAAI/bge-m3", min_length=1)
    revision: str | None = None
    batch_size: int = Field(default=32, ge=1)
    cache_entries: int = Field(
        default=10_000,
        ge=0,
        description="Embedding cache size, in vectors. ``0`` disables it. Keyed by the "
        "canonical ``EmbedFingerprint`` — not by the model's name, which carries neither the "
        "pooling nor the revision, so a name-keyed cache would serve vectors from the previous "
        "space to a re-embed and report success. Changing any identity field makes the old "
        "entries unreachable rather than stale, so there is no flush step to forget.",
    )


class LlmSettings(Section):
    """The generation runtime. Local and hosted differ by ``base_url`` and nothing else."""

    provider: str = "ollama"
    model: str = Field(default="qwen2.5:14b", min_length=1)
    base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)


class RagSettings(Section):
    """Retrieval and chunking."""

    profile: RetrievalProfile = RetrievalProfile.BALANCED
    chunker: str = Field(default="structural", description="Registered chunker to use.")
    pipeline: tuple[str, ...] = Field(
        default=("dense", "lexical", "rrf"),
        min_length=1,
        description="Retrieval stages, in order. A pipeline is declared here rather than "
        "assembled in code, so two of them can be compared by configuration alone.",
    )
    reranker: str | None = Field(
        default=None,
        description="Reranker to append when the profile asks for one. ``None`` disables "
        "reranking regardless of profile.",
    )
    overrides: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Per-field overrides on the selected profile. Everything not named here "
        "keeps the profile's value.",
    )


class ConnectorSettings(Section):
    """One configured source."""

    type: str = Field(min_length=1, description="Registered connector implementation.")
    enabled: bool = True
    schedule_s: int | None = Field(
        default=None, ge=60, description="Poll interval. ``None`` means manual sync only."
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict, description="Validated against the connector's own config model."
    )


class PluginSettings(Section):
    enabled: tuple[str, ...] | None = Field(
        default=None,
        description="If set, only these plugins load. ``None`` loads everything installed — "
        "discovery finds plugins, and this filters them.",
    )
    disabled: tuple[str, ...] = ()
    middleware: tuple[str, ...] = Field(
        default=(),
        description="Middleware to run, in this order. Order is declared where a reader can "
        "see it rather than emerging from priority numbers spread across packages.",
    )
    config: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description="Per-component configuration, keyed ``<kind>.<name>`` — for example "
        "``parser.pdf``. Validated against that component's declared model.",
    )
    registry_url: str = Field(
        default="https://raw.githubusercontent.com/mgd43b/manicule/main/community-plugins.json",
        description="Browsable list of community plugins.",
    )
    allow_install: bool = Field(
        default=False,
        description="Whether plugin installation is offered in-product. Off by default: "
        "installing a plugin runs its code with this process's full authority.",
    )


# --- root ------------------------------------------------------------------------------------


class Settings(BaseSettings):
    """Everything manicule reads at startup."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        nested_model_default_partial_update=True,
        secrets_dir=None,
        env_file_encoding="utf-8",
    )

    workspace: str = Field(default="default", min_length=1)
    mode: Mode = Mode.PERSONAL
    locale: str = "auto"
    data_dir: Path = Field(default_factory=default_data_dir)
    cache_dir: Path = Field(default_factory=default_cache_dir)

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    rag: RagSettings = Field(default_factory=RagSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    connectors: dict[str, ConnectorSettings] = Field(default_factory=dict)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    parser_fallbacks: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Ordered parser chains by media type. ``*`` supplies a global tail. Every "
        "named parser must be installed — a chain that varies by machine chunks the same "
        "document differently depending on where it was ingested.",
    )
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    events: EventSettings = Field(default_factory=EventSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    ui: UiSettings = Field(default_factory=UiSettings)

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Layer the sources, highest priority first."""
        del dotenv_settings  # replaced, so that the files are chosen at load time not import
        return (
            init_settings,
            env_settings,
            PrefixedDotEnvSource(
                settings_cls,
                env_file=env_files(),
                env_file_encoding="utf-8",
                env_prefix=ENV_PREFIX,
                env_nested_delimiter="__",
            ),
            TomlConfigSettingsSource(settings_cls, toml_file=config_file()),
            file_secret_settings,
        )

    @field_validator("data_dir", "cache_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()

    @override
    def model_post_init(self, context: Any, /) -> None:
        """Fill provider credentials from the environment by convention."""
        del context
        self.providers = resolve_provider_keys(
            self.providers, self.selected_providers, environ=provider_environment()
        )

    # --- derived ------------------------------------------------------------------------

    @property
    def selected_providers(self) -> frozenset[str]:
        """Providers this configuration actually uses."""
        return frozenset({self.llm.provider.lower(), self.embedding.provider.lower()})

    @property
    def cloud_providers_in_use(self) -> frozenset[str]:
        """Selected providers that send content off this machine."""
        return frozenset(name for name in self.selected_providers if not is_local(name))

    def provider(self, name: str) -> ProviderSettings:
        """Settings for one provider, defaulted if never configured."""
        return self.providers.get(name.lower(), ProviderSettings())

    def component_config(self, kind: str, name: str) -> Mapping[str, JsonValue]:
        """Raw configuration for one component, before validation."""
        return self.plugins.config.get(f"{kind}.{name}", {})

    def redacted(self) -> dict[str, JsonValue]:
        """The configuration with every secret replaced by a placeholder.

        This is what ``config show`` and the configuration API return. Returning the live
        object instead would hand out every API key, OAuth client secret and webhook signing
        key to anyone allowed to read configuration.
        """
        dumped: Any = self.model_dump(mode="json")
        return _mask(dumped)

    # --- policy -------------------------------------------------------------------------

    def policy_problems(self) -> list[str]:
        """Configurations that are individually valid and jointly wrong.

        Checked once at startup, before anything is constructed, so an impossible setup fails
        immediately instead of at the first request that happens to exercise it.
        """
        problems: list[str] = []

        cloud = self.cloud_providers_in_use
        if cloud and not self.security.data_policy.cloud_allowed:
            names = ", ".join(sorted(cloud))
            problems.append(
                f"security.data_policy.cloud_allowed is false, but the hosted provider(s) "
                f"{names} are selected. Choose a local provider, or allow cloud processing."
            )

        for name in sorted(self.selected_providers):
            if not is_local(name) and not self.provider(name).has_key:
                expected = " or ".join(env_var_names(name))
                problems.append(
                    f"provider {name!r} is selected but has no API key. Set {expected}, or "
                    f"providers.{name}.api_key."
                )

        transport = self.security.transport
        if not transport.is_loopback and self.security.auth.mode is AuthMode.NONE:
            problems.append(
                f"security.transport.bind_host is {transport.bind_host!r} with "
                f"security.auth.mode 'none'. An unauthenticated index on a routable address "
                f"is readable by anyone who can reach it. Bind 127.0.0.1, or enable auth."
            )

        if self.security.auth.mode is AuthMode.OAUTH and not self.security.auth.providers:
            problems.append("security.auth.mode is 'oauth' but no OAuth providers are configured")

        if self.security.audit.destination is AuditDestination.WEBHOOK and not self.events.webhooks:
            problems.append("security.audit.destination is 'webhook' but events.webhooks is empty")

        return problems

    def require_valid(self) -> Self:
        """Raise if this configuration cannot be run.

        Raises:
            PolicyError: With every problem listed, not just the first — fixing one
                misconfiguration only to be told about the next is a poor way to spend an
                afternoon.
        """
        problems = self.policy_problems()
        if problems:
            joined = "\n  - ".join(problems)
            msg = f"configuration cannot be run:\n  - {joined}"
            raise PolicyError(msg)
        return self


_SECRET_KEYS = ("api_key", "secret", "token", "password", "client_secret", "encryption_key")


def looks_secret(key: str) -> bool:
    """Whether a field name identifies a credential.

    One rule, used both to mask configuration for display and to omit it when writing.
    """
    normalised = key.replace("-", "").replace("_", "").lower()
    return any(marker.replace("_", "") in normalised for marker in _SECRET_KEYS)


def _mask(value: JsonValue, key: str = "") -> Any:  # noqa: ANN401 - recursive over JSON
    if isinstance(value, dict):
        return {k: _mask(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, key) for v in value]
    if value is not None and looks_secret(key):
        return "**********"
    return value


__all__ = [
    "APP_NAME",
    "ENV_PREFIX",
    "AtRestSettings",
    "AuditDestination",
    "AuditSettings",
    "AuthMode",
    "AuthSettings",
    "ConnectorSettings",
    "DataPolicySettings",
    "EmbeddingSettings",
    "EventSettings",
    "IngestSettings",
    "LlmSettings",
    "Mode",
    "OAuthProvider",
    "PluginSettings",
    "ProviderSettings",
    "RagSettings",
    "RedactionMethod",
    "RedactionSettings",
    "Role",
    "SecuritySettings",
    "Settings",
    "SourceRestrictions",
    "StorageSettings",
    "TelemetrySettings",
    "Theme",
    "TransportSettings",
    "UiSettings",
    "WebhookSettings",
    "WorkspaceOverride",
    "config_file",
    "default_cache_dir",
    "default_config_dir",
    "default_data_dir",
    "env_files",
    "looks_secret",
    "provider_environment",
]
