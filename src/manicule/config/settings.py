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
from ipaddress import ip_network
from pathlib import Path
from typing import Any, Literal, Self, override
from urllib.parse import urlsplit

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
    Endpoint,
    ModelRole,
    ProviderSettings,
    egress_for,
    env_var_names,
    needs_credential,
    resolve_provider_keys,
    runs_in_process,
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


class RedactionScope(StrEnum):
    """How much of what a model is given gets redacted."""

    REMOTE = "remote"
    """Only when the resolved endpoint leaves this machine.

    The point of the feature: what leaves is redacted, what stays is not, so a fully local
    install pays nothing for a threat it does not have.
    """

    ALWAYS = "always"
    """Regardless of egress.

    For the one case classification cannot see — a proxy on loopback that forwards to a
    hosted provider (:class:`~manicule.config.providers.Egress`) — and for operators who want
    the model's *input* uniform, which is also the only way a local and a hosted deployment
    produce comparable answers.
    """


class RedactionSettings(Section):
    """Personal-data redaction.

    Applied where the outbound context is assembled, so it governs what is *sent to a model*
    and leaves the index intact. Redacting at ingest would destroy the stored document
    permanently while doing nothing about what the model later sees, which is the opposite of
    what the setting's name promises.

    Off by default, and that is a decision with a stated cost rather than an oversight.
    Detectors are recall-oriented and will fire on things that are not personal data — a
    version string that looks like a phone number, an internal identifier that looks like a
    card — and a model that cannot see the address cannot answer a question about it.
    """

    enabled: bool = False
    scope: RedactionScope = RedactionScope.REMOTE
    patterns: tuple[str, ...] = Field(
        default=(),
        description="Named detectors, e.g. ``email``, ``phone``, ``credit-card``, "
        "``ip-address``. Named rather than written as raw regexes so that a config file is a "
        "policy rather than a program, and so the detectors can be tested and improved.",
    )
    custom_patterns: tuple[str, ...] = Field(
        default=(),
        description="Additional regexes, compiled at startup. One that does not compile is a "
        "refusal naming the pattern and the error — a silently dropped pattern makes "
        "redaction weaker than the configuration says it is.",
    )
    method: RedactionMethod = RedactionMethod.REPLACE
    replacement: str = "[REDACTED]"
    hash_salt: SecretStr | None = Field(
        default=None,
        description="Per-installation secret for ``method = 'hash'``, generated on first use "
        "and never sent anywhere. An *unsalted* digest of an email address is reversible by "
        "anyone with a word list, so sending one instead of the value would be privacy "
        "theatre that costs answer quality and buys nothing.",
    )
    timeout_s: float = Field(
        default=5.0,
        gt=0,
        description="Wall clock for redacting one request. A regex over 32k tokens of "
        "context with operator-supplied patterns is a denial-of-service surface and Python's "
        "``re`` cannot be interrupted, so this runs in a worker thread under a deadline. "
        "**Exceeding it fails the query** — the fail-safe direction is refuse-to-send.",
    )


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
    allowed_origins: tuple[str, ...] = Field(
        default=(),
        description="Origins the browser API may be called from, in full ``scheme://host[:port]`` "
        "form. Empty means same-origin only. There is deliberately no wildcard: an index of "
        "somebody's documents readable from any page they visit is not a default.",
    )
    allowed_endpoints: tuple[str, ...] = Field(
        default=(),
        description="Outbound hosts plugins and connectors may reach. Empty is unrestricted.",
    )
    widget_allowed_domains: tuple[str, ...] = Field(
        default=(),
        description="Origins permitted to embed the chat widget in a frame. Empty means none, "
        "so the default answer to a framing attempt is a refusal rather than a click nobody "
        "meant to make.",
    )

    @field_validator("trusted_proxies")
    @classmethod
    def _proxies_are_networks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Every entry must parse as a network, at startup.

        A typo here does not fail loudly on its own: the entry simply never matches, the
        proxy in front is never trusted, and the deployment quietly attributes every request
        to the proxy's own address. That is the safe direction and it is still wrong, because
        the operator believes a policy is in force that is not. So it is refused.

        Raises:
            ValueError: An entry is not an address or a CIDR range.
        """
        for entry in value:
            text = entry.strip()
            if not text:
                msg = "security.transport.trusted_proxies contains an empty entry"
                raise ValueError(msg)
            try:
                # `strict=False`: a bare host address means that single host.
                ip_network(text, strict=False)
            except ValueError as exc:
                msg = (
                    f"security.transport.trusted_proxies entry {entry!r} is not an address or "
                    f"a CIDR range ({exc}). This list decides whose X-Forwarded-For header is "
                    f"believed, so an entry that silently matches nothing is refused."
                )
                raise ValueError(msg) from exc
        return value

    @field_validator("allowed_origins")
    @classmethod
    def _origins_are_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """No wildcard, and every entry is a real origin.

        ``*`` is refused rather than accepted-and-ignored. An operator who writes it has asked
        for "any page on the internet may read this index with the browser's credentials", and
        the honest response is to say that is not offered rather than to silently narrow it.

        Raises:
            ValueError: An entry is ``*`` or is not a ``scheme://host[:port]`` origin.
        """
        for entry in value:
            if entry.strip() == "*":
                msg = (
                    "security.transport.allowed_origins may not contain '*'. A cross-origin "
                    "wildcard over a document index means any page a user visits can read it; "
                    "list the origins that may embed or call this installation."
                )
                raise ValueError(msg)
            parsed = urlsplit(entry)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path:
                msg = (
                    f"security.transport.allowed_origins entry {entry!r} is not an origin. "
                    f"Write it as scheme://host[:port], with no path."
                )
                raise ValueError(msg)
        return value

    @property
    def is_loopback(self) -> bool:
        return self.bind_host in {"127.0.0.1", "::1", "localhost"}


class AtRestSettings(Section):
    redact_logs_content: bool = Field(default=True, description="Keep document text out of logs.")


class SharingSettings(Section):
    """Shared conversation links.

    A share link is a **bearer capability for an unauthenticated URL**, so every setting here
    bounds one: whether links can be minted at all, and how long a minted one lives.
    """

    enabled: bool = Field(
        default=True,
        description="One switch rather than a per-field disclosure policy nobody configures "
        "correctly. A document *title* can itself be sensitive, and an anonymous viewer sees "
        "titles, so a deployment that cannot disclose those turns sharing off entirely.",
    )
    link_ttl_s: int = Field(
        default=30 * 24 * 3600,
        gt=0,
        description="How long a share link stays valid. A capability with no expiry "
        "accumulates forever and the set of live ones becomes unknowable.",
    )


class SecuritySettings(Section):
    auth: AuthSettings = Field(default_factory=AuthSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    data_policy: DataPolicySettings = Field(default_factory=DataPolicySettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    storage: AtRestSettings = Field(default_factory=AtRestSettings)
    sharing: SharingSettings = Field(default_factory=SharingSettings)


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

    provider: str = Field(
        default="mlx", min_length=1, description="Which embedder implementation to use."
    )
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
    """The generation runtime. Local and hosted differ by ``base_url`` and nothing else.

    **There are three timeouts, because one covers the wrong interval.** A single budget
    around the call that returns a stream bounds time-to-first-byte and nothing after it, so
    a provider that opens a stream and then stops sending blocks forever and is
    indistinguishable from a slow answer.
    """

    provider: str = Field(
        default="ollama",
        min_length=1,
        description="Which **vendor** serves the model: ``ollama``, ``openai``, "
        "``anthropic``. This is what decides which credential is needed and whether the "
        "endpoint leaves this machine — not which component is built. One dependency speaks "
        "to all of them, and it is named by ``generator``.",
    )
    generator: str = Field(
        default="litellm",
        min_length=1,
        description="Which registered generator **component** to build. Separate from "
        "``provider`` because the two answer different questions and conflating them made "
        "the default configuration unrunnable: one implementation reaches every vendor "
        "through a base_url, so the component is not a function of the vendor. Change this "
        "only to select a third-party generator.",
    )
    model: str = Field(default="qwen2.5:14b", min_length=1)
    base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="What the model may produce **and** the ``generation_reserve`` term of "
        "the startup window cross-check. Deliberately one number: two numbers for one "
        "quantity disagree by default, and then ``finish_reason='length'`` stops meaning "
        "anything precise.",
    )
    timeout_s: float = Field(
        default=120.0, gt=0, description="Total wall clock for one generation."
    )
    first_token_timeout_s: float = Field(
        default=60.0,
        gt=0,
        description="Connect, queue, prompt evaluation and model load. Generous because a "
        "cold local model is loaded into memory first, which is a real multi-second cost the "
        "first time.",
    )
    stream_idle_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="The gap between two tokens. This is what turns a hung provider into an "
        "error rather than an answer that never finishes.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries — not attempts — of the *connection*, before the first token. "
        "After the first token a failure is terminal: restarting makes the reader watch the "
        "answer rewind, and continuing splices two independently-sampled answers into text "
        "no single generation produced.",
    )
    keep_alive: str = Field(
        default="10m",
        description="How long a served local model stays resident. A pure throughput knob: "
        "Ollama unloads an idle model after five minutes and the next question then pays a "
        "multi-second load from disk. It changes nothing about any answer.",
    )
    context_window: int | None = Field(
        default=None,
        ge=1,
        description="Override for the window that will actually be **served**. Normally "
        "determined from the runtime — Ollama's ``/api/show`` combined with the ``num_ctx`` "
        "manicule sets, or the library's model metadata for a hosted provider. Set this only "
        "for an endpoint whose model neither can describe, such as an OpenAI-compatible "
        "server with a private model name; without it such a configuration is refused at "
        "startup rather than left to discover the limit by exceeding it.",
    )
    citation_verify_timeout_s: float = Field(
        default=5.0,
        gt=0,
        description="Budget for verifying citations, measured from the start of the answer. "
        "Generous because the work starts before the first token. A marker whose "
        "verification has not finished when it must be emitted is **dropped**, with its own "
        "reason: sending an unverified citation under a design whose whole claim is "
        "verification is the unacceptable half of that trade.",
    )
    token_safety_factor: float = Field(
        default=1.15,
        ge=1.0,
        description="How much the prompt estimate is inflated. Biased toward overcounting "
        "because the errors are not symmetric: undercounting overflows the window and gets "
        "the context truncated by the server, which is the silent failure, while "
        "overcounting costs a passage. **Never auto-tuned** — an estimator that adapts makes "
        "two runs non-comparable, and it adapts in the unsafe direction after a run of short "
        "answers. A diagnostic recommends a value; a human sets it.",
    )
    token_drift_tolerance: float = Field(
        default=0.15,
        ge=0.0,
        description="Relative disagreement between the estimate and the provider's true "
        "prompt count that is treated as ordinary tokenizer drift. Beyond it, an error-level "
        "event naming both numbers and the model.",
    )
    system_prompt_extra: str = Field(
        default="",
        description="Instructions appended to the system prompt. Appended, never "
        "substituted: the citation protocol is not configurable, because the binder's "
        "guarantees assume the model was told it. Counted into the startup window "
        "cross-check, so a long custom prompt is refused rather than silently displacing "
        "passages.",
    )


class QueryCacheSettings(Section):
    """The L1 query-result cache: ranked chunk ids, never chunk text.

    Caching the *decision* rather than the content is what makes a hit incapable of serving a
    soft-deleted, unindexed or foreign-workspace chunk: the entry holds no chunks, so the
    boundary is re-enforced on every hit through the same join the dense leg uses rather than
    snapshotted at the moment of the miss.
    """

    enabled: bool = Field(
        default=True,
        description="Turned off for evaluation runs, which must measure retrieval rather than "
        "the cache. A flag, not a code path — the pipeline is identical either way.",
    )
    entries: int = Field(
        default=512, ge=0, description="Rankings held, least-recently-used evicted. ``0`` is off."
    )
    ttl_s: float = Field(
        default=300.0,
        gt=0,
        description="A bound on staleness from anything the generation counter was not taught "
        "about. Belt-and-braces underneath the counter, never the mechanism.",
    )


class RouterSettings(Section):
    """The deterministic query router: a pure function over the query text.

    Tuned for precision rather than recall, and the rule is worth stating because it decides
    every question about the pattern lists: **a missed greeting costs one retrieval, which is
    harmless; a false greeting costs a wrong answer to a real question, which is not.** When
    in doubt, retrieve.
    """

    enabled: bool = True
    max_chars: int = Field(
        default=40,
        ge=1,
        description="Longest input that may route away from the corpus. A greeting is short; "
        "a sentence beginning with one is a question.",
    )
    greetings: tuple[str, ...] = Field(
        default=(
            "hi",
            "hello",
            "hey",
            "howdy",
            "yo",
            "sup",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "thanks!",
            "cheers",
            "bye",
            "goodbye",
        ),
        description="Whole inputs that are greetings, matched in full and never as a prefix. "
        "Configuration rather than a constant, because any list is incomplete on a "
        "multilingual corpus and being incomplete costs only latency.",
    )


class ContextSettings(Section):
    """How the assembled context is measured against the generator's window.

    The tokenizer here is **not** the one that sized the chunks. That budget is measured in the
    embedder's vocabulary, to stop the embedder truncating silently; this one is measured in
    the generator's, to stop the server truncating the prompt. ``Chunk.token_count`` is the
    first of those and is wrong for this purpose by an unknown factor.
    """

    encoding: str = Field(
        default="o200k_base",
        min_length=1,
        description="A ``tiktoken`` encoding name, never a model name. The generator is "
        "Ollama-hosted and runs a Llama, Qwen or Mistral vocabulary — none of them tiktoken's "
        "— so naming a model here would make an estimate look authoritative.",
    )
    safety_factor: float = Field(
        default=1.2,
        ge=1.0,
        description="Inflation applied to the estimate. Undercounting overflows the window and "
        "the server truncates the prompt, silently; overcounting costs a passage. The error is "
        "pushed in the direction that is visible.",
    )
    drift_tolerance: float = Field(
        default=0.15,
        ge=0.0,
        description="How far the estimate may sit from the generator's own "
        "``prompt_eval_count`` before it is an error worth surfacing. Measuring once beats a "
        "safety factor forever.",
    )
    system_prompt_tokens: int = Field(
        default=400,
        ge=0,
        description="Room the citation protocol and system prompt occupy, for the startup "
        "cross-check against the generator's window.",
    )


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
    cache: QueryCacheSettings = Field(default_factory=QueryCacheSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    assert_scope: bool = Field(
        default=False,
        description="Run the pipeline's scope assertion on every query, as a runtime check "
        "rather than only in the suite. Off by default because it costs a document lookup per "
        "candidate per stage; on, it holds a live pipeline to the property that makes the "
        "vector store's ``workspace_ids`` exemption safe.",
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
    def selected_endpoints(self) -> tuple[Endpoint, ...]:
        """Every model endpoint this configuration will open, with where its content goes.

        This is the resolution :func:`~manicule.config.providers.egress_for` needs and cannot
        do on its own, because the endpoint is spread across two places: ``llm.base_url``
        overrides for generation, and ``providers.<name>.base_url`` — already defaulted by
        :func:`~manicule.config.providers.resolve_provider_keys` — covers both roles.

        Per role rather than per provider name, because the same provider can be configured
        at two different addresses and only one of them may be on this machine.
        """
        return (
            self._endpoint(ModelRole.LLM, self.llm.provider, self.llm.base_url),
            self._endpoint(ModelRole.EMBEDDING, self.embedding.provider, None),
        )

    def _endpoint(self, role: ModelRole, provider: str, override: str | None) -> Endpoint:
        name = provider.strip().lower()
        base_url = override or self.provider(name).base_url
        return Endpoint(
            role=role,
            provider=name,
            base_url=base_url,
            egress=egress_for(name, base_url),
        )

    @property
    def cloud_providers_in_use(self) -> frozenset[str]:
        """Selected providers whose endpoint is off this machine.

        Derived from the endpoints rather than from the provider names: an ``ollama`` at
        ``http://gpu-box.lan:11434`` belongs here, and an OpenAI-compatible server on
        ``127.0.0.1`` does not.
        """
        return frozenset(
            endpoint.provider for endpoint in self.selected_endpoints if endpoint.leaves_machine
        )

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

        if not self.security.data_policy.cloud_allowed:
            for endpoint in self.selected_endpoints:
                if endpoint.leaves_machine:
                    problems.append(
                        f"security.data_policy.cloud_allowed is false, but the "
                        f"{endpoint.describe()} is not on this machine, so every prompt and "
                        f"every retrieved passage would cross the network to reach it. Point "
                        f"it at loopback, choose an in-process provider, or allow cloud "
                        f"processing."
                    )

        for endpoint in self.selected_endpoints:
            if runs_in_process(endpoint.provider) and endpoint.base_url:
                problems.append(
                    f"the {endpoint.describe()} carries a base_url, but "
                    f"{endpoint.provider!r} runs in this process and dials nothing. That "
                    f"setting is not in force; remove it, or select a served provider."
                )

        for name in sorted(self.selected_providers):
            if needs_credential(name) and not self.provider(name).has_key:
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

        problems.extend(self._redaction_problems())
        problems.extend(self._source_restriction_problems())

        if self.security.auth.mode is AuthMode.OAUTH and not self.security.auth.providers:
            problems.append("security.auth.mode is 'oauth' but no OAuth providers are configured")

        if self.security.audit.destination is AuditDestination.WEBHOOK and not self.events.webhooks:
            problems.append("security.audit.destination is 'webhook' but events.webhooks is empty")

        return problems

    def _redaction_problems(self) -> list[str]:
        """Redaction settings that cannot do what they say.

        A custom pattern that does not compile is the case worth naming: swallowed, it makes
        redaction quietly weaker than the configuration claims, which is precisely the
        "appears to be in force and silently is not" failure this project refuses.
        """
        import re  # noqa: PLC0415 - only this check needs it

        from manicule.generation.redaction import BUILTIN_DETECTORS  # noqa: PLC0415

        redaction = self.security.data_policy.auto_redact
        problems: list[str] = []
        unknown = sorted(
            name for name in redaction.patterns if name.strip().lower() not in BUILTIN_DETECTORS
        )
        if unknown:
            available = ", ".join(sorted(BUILTIN_DETECTORS))
            problems.append(
                f"security.data_policy.auto_redact.patterns names {', '.join(unknown)}, which "
                f"is not a built-in detector. Available: {available}. Put a regex of your own "
                f"in custom_patterns instead."
            )
        for pattern in redaction.custom_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                problems.append(
                    f"security.data_policy.auto_redact.custom_patterns contains "
                    f"{pattern!r}, which is not a valid regular expression: {exc}. Fix it or "
                    f"remove it; a pattern that does not compile cannot redact anything, and "
                    f"dropping it silently would make redaction weaker than this file says."
                )
        if redaction.enabled and not (redaction.patterns or redaction.custom_patterns):
            problems.append(
                "security.data_policy.auto_redact.enabled is true with no patterns and no "
                "custom_patterns, so nothing would be redacted while the setting reads as on. "
                "Name at least one detector, or set enabled = false."
            )
        return problems

    def _source_restriction_problems(self) -> list[str]:
        """Per-source policies that contradict each other.

        ``local_only`` is a floor and ``cloud_allowed`` is an exemption, so a source named in
        both asks for two incompatible things. Resolving it silently — either way — means one
        of the two settings is not in force and nothing says which.
        """
        restrictions = self.security.data_policy.source_restrictions
        problems: list[str] = []
        both = sorted(set(restrictions.local_only) & set(restrictions.cloud_allowed))
        if both:
            problems.append(
                f"security.data_policy.source_restrictions names {', '.join(both)} in both "
                f"local_only and cloud_allowed. local_only is a floor that no exemption "
                f"releases, so one of the two settings would not be in force. Remove the "
                f"source from whichever list is wrong."
            )

        # A workspace override is only ever consulted by exact name, so a key naming no
        # workspace is a restriction that reads as in force and is not — and the direction it
        # fails in is permissive.
        # Every workspace that can actually ask a question — this installation's own, and
        # any an OAuth provider places its users in. Checking against the former alone made a
        # shipped multi-workspace configuration unrunnable, and the refusal's own premise
        # ("a workspace that never asks a question") was false in exactly that setup.
        reachable = {self.workspace.strip().lower()} | {
            provider.workspace.strip().lower()
            for provider in self.security.auth.providers
            if provider.workspace
        }
        stray = sorted(
            name
            for name in self.security.data_policy.workspace_overrides
            if name.strip().lower() not in reachable
        )
        if stray:
            known = ", ".join(sorted(reachable))
            problems.append(
                f"security.data_policy.workspace_overrides names {', '.join(stray)}, which no "
                f"workspace on this installation uses. Reachable workspaces: {known}. An "
                f"override keyed to a workspace that never asks a question is never applied, "
                f"and a restriction that is not applied reads as in force and is not."
            )
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
    "ContextSettings",
    "DataPolicySettings",
    "EmbeddingSettings",
    "EventSettings",
    "IngestSettings",
    "LlmSettings",
    "Mode",
    "OAuthProvider",
    "PluginSettings",
    "ProviderSettings",
    "QueryCacheSettings",
    "RagSettings",
    "RedactionMethod",
    "RedactionScope",
    "RedactionSettings",
    "Role",
    "RouterSettings",
    "SecuritySettings",
    "Settings",
    "SharingSettings",
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
