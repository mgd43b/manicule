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
from collections.abc import Mapping, Sequence
from enum import StrEnum
from ipaddress import ip_network
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, get_args, get_origin, override
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
    CLI_AUTH_PROVIDERS,
    Endpoint,
    ModelRole,
    ProviderSettings,
    egress_for,
    endpoint_egress,
    env_var_names,
    needs_credential,
    resolve_provider_keys,
    runs_in_process,
)
from manicule.core.acquisition import SnapshotPromotionPolicy
from manicule.core.ann import MINIMUM_ANN_INDEX_THRESHOLD
from manicule.core.embedding import PrefixScheme
from manicule.core.errors import PolicyError
from manicule.core.retrieval import RetrievalProfile

ENV_PREFIX = "MANICULE_"
APP_NAME = "manicule"

QDRANT_COLLECTION_PREFIX = "manicule"
"""What this installation's Qdrant collections are called before its own scopes are appended.

The default lives here rather than beside the store, because a default that also exists in the
implementation is a second answer to one question, and the two only disagree once somebody
changes the wrong one. The store takes the prefix it is given and has no opinion about it.
"""

QDRANT_API_KEY_ENV = "QDRANT_API_KEY"
"""The conventional variable a Qdrant key arrives in, read when configuration sets none.

The same convention model providers use — ``<SERVICE>_API_KEY`` — because an operator who has
already exported it for `qdrant`'s own tooling should not have to write it down twice, and a
credential in a config file is a credential in a backup.
"""


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
    """Where regenerable artifacts live. Safe to delete."""
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
    whatever else the project needs. Treating every unrecognized line as a misspelled
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
    """Whether this installation serves one person or a team.

    The difference is who a caller *without* a credential is. In ``personal`` mode, with
    ``security.auth.mode = none``, it is the operator at this machine, holding the authority the
    command line already gives them. In ``team`` mode there is no such person: several people
    share the installation, so every caller must present a credential and a served process with
    authentication off is refused — ``--no-authentication`` included, because an anonymous
    administrator is a single-operator arrangement by definition.
    """

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


class LoggingSettings(Section):
    requests: bool = Field(
        default=True,
        description="Write content-free HTTP and MCP request summaries to a local file and stderr.",
    )
    file: Path = Field(
        default=Path("logs/requests.jsonl"),
        description="Request log file. Relative paths are resolved beneath data_dir.",
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024, ge=1, description="Rotate the request log at this many bytes."
    )
    backup_count: int = Field(
        default=5, ge=1, description="Number of rotated request logs to keep."
    )


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
        "theater that costs answer quality and buys nothing.",
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
    """One identity provider a person may sign in through.

    Signing in admits a person to **one** workspace, in the role named here, the first time
    they arrive; after that their role is whatever an administrator has made it. Admission is
    re-checked on every sign-in rather than only the first, so removing an address from
    ``allowed_emails`` stops that person at their next sign-in instead of never.
    """

    type: Literal["google", "github"]
    client_id: str = Field(min_length=1)
    client_secret: SecretStr
    redirect_uri: str | None = Field(
        default=None,
        description="The callback registered with the provider, "
        "``https://<host>/auth/callback/<type>``. Required to serve: deriving it from the "
        "request would let a ``Host`` header choose where the provider sends the code.",
    )
    workspace: str | None = Field(
        default=None,
        description="The workspace this provider admits people to. Unset means whichever "
        "workspace the process serves; set, the provider is offered only by a process serving "
        "that workspace.",
    )
    role: Role = Field(
        default=Role.MEMBER, description="The role a person holds the first time they sign in."
    )
    allowed_emails: tuple[str, ...] = Field(
        default=(), description="Verified addresses admitted by exact, case-insensitive match."
    )
    allowed_domains: tuple[str, ...] = Field(
        default=(),
        description="Domains whose verified addresses are admitted, e.g. ``example.org``. "
        "Exact match on the part after the ``@`` — a subdomain is a different domain.",
    )
    allow_any_user: bool = Field(
        default=False,
        description="Admit every account the provider will authenticate. For GitHub that is "
        "anybody on the internet with an account, which is why it is its own switch rather "
        "than what an empty allowlist means.",
    )

    @field_validator("allowed_emails", "allowed_domains")
    @classmethod
    def _entries_are_normalized(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lower-cased and stripped, and a blank entry refused.

        Compared case-insensitively at sign-in, so the stored form is the compared form: an
        allowlist that held ``Alice@Example.org`` and matched ``alice@example.org`` only
        because a comparison remembered to fold case is one refactor from not matching it.

        Raises:
            ValueError: An entry is empty, or a domain entry carries an ``@``.
        """
        normalized: list[str] = []
        for entry in value:
            text = entry.strip().lower()
            if not text:
                msg = "an OAuth provider allowlist contains an empty entry"
                raise ValueError(msg)
            normalized.append(text)
        return tuple(normalized)

    @field_validator("allowed_domains")
    @classmethod
    def _domains_are_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A domain entry is a domain, not an address and not a pattern.

        Raises:
            ValueError: An entry carries ``@`` or ``*``.
        """
        for entry in value:
            if "@" in entry or "*" in entry:
                msg = (
                    f"allowed_domains entry {entry!r} is not a domain. Write the part after "
                    f"the '@' exactly (example.org); an address belongs in allowed_emails, and "
                    f"there is no wildcard."
                )
                raise ValueError(msg)
        return value

    @property
    def admits_anybody(self) -> bool:
        """Whether any sign-in could ever be admitted through this provider."""
        return self.allow_any_user or bool(self.allowed_emails) or bool(self.allowed_domains)


class AuthSettings(Section):
    mode: AuthMode = Field(
        default=AuthMode.NONE,
        description="``none`` is only permitted while every interface is bound to loopback.",
    )
    providers: tuple[OAuthProvider, ...] = ()
    session_secret: SecretStr | None = Field(
        default=None,
        description="Key signing the browser's session and sign-in cookies, at least 32 "
        "characters. Required when ``mode = 'oauth'``. Changing it signs every browser out, "
        "which is also how to do that deliberately.",
    )
    session_max_age_s: int = Field(
        default=60 * 60 * 24 * 7,
        ge=60,
        description="How long a browser session lasts from sign-in. Not extended by use: a "
        "session that renews itself on every request never ends for anybody who keeps a tab "
        "open.",
    )


class RateLimitSettings(Section):
    """The in-process token bucket in front of every network surface.

    One bucket per caller — the API key, the signed-in person, or, for a caller presenting
    neither, the client address as :class:`~manicule.api.proxy.ProxyPolicy` resolves it. A
    separate, much smaller bucket per address meters **failed** authentication: a header
    credential that did not work is charged to it, and once it is spent such requests are
    refused with 429. A credential that works is never refused by it, whoever shares the address.
    """

    enabled: bool = True
    per_minute: int = Field(
        default=600, ge=1, description="Sustained requests per minute for one caller."
    )
    burst: int = Field(default=120, ge=1, description="Requests one caller may make at once.")
    failed_auth_per_minute: int = Field(
        default=10,
        ge=1,
        description="Failed authentications per minute one address may make. Past it, "
        "requests from that address presenting a credential that does not work are refused "
        "with 429 until the bucket refills; a working credential from the same address never "
        "is.",
    )
    max_tracked: int = Field(
        default=10_000,
        ge=100,
        description="Callers remembered at once. The least recently seen is forgotten first, "
        "so memory is bounded however many addresses arrive.",
    )


class AlertSettings(Section):
    """Security alerts: patterns in how the installation is being used, not single events.

    An alert is recorded in the workspace's alert list, logged at warning, and audited when
    auditing is on. Each fires at most once per subject per window, so a sustained attack
    produces one alert per window rather than one per request.
    """

    enabled: bool = True
    window_s: int = Field(
        default=300, ge=10, description="The sliding window every threshold is counted over."
    )
    failed_auth_threshold: int = Field(
        default=20,
        ge=1,
        description="Failed authentications from one address within the window: brute force.",
    )
    key_address_threshold: int = Field(
        default=5,
        ge=2,
        description="Distinct client addresses presenting one API key within the window: a key "
        "that has leaked, or is being shared.",
    )
    export_document_threshold: int = Field(
        default=500,
        ge=1,
        description="Documents whose content one caller reads within the window: an export "
        "happening through an interface that was not meant for one.",
    )


class TransportSettings(Section):
    """How manicule is reachable over the network."""

    bind_host: str = Field(
        default="127.0.0.1",
        description="Loopback by default. An unauthenticated service on a routable address "
        "is an open document index, so binding wider requires authentication to be on.",
    )
    port: int = Field(default=8765, ge=1, le=65535)
    enforce_https: bool = Field(
        default=True,
        description="Browser credentials require HTTPS: the session and sign-in cookies carry "
        "``Secure``, and an OAuth ``redirect_uri`` must be ``https://`` unless it names a "
        "loopback host. Off only for a plain-HTTP network an operator owns, where a cookie "
        "marked ``Secure`` would never be sent back.",
    )
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

        **Anything past the port is refused too** — a path, a query string, a fragment, a
        trailing slash. A browser's ``Origin`` header is exactly scheme, host and port, so an
        entry carrying any of those can never match one: CORS is silently off for that origin
        while the configuration file says it is on. That is the failure mode this whole module
        is written against, and it is quieter here than most, because the symptom appears as a
        browser error on somebody else's page rather than anywhere an operator is looking.

        Raises:
            ValueError: An entry is ``*`` or is not exactly a ``scheme://host[:port]`` origin.
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
            extra = parsed.path or parsed.query or parsed.fragment
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or extra:
                msg = (
                    f"security.transport.allowed_origins entry {entry!r} is not an origin. "
                    f"Write it as scheme://host[:port] — nothing after the port, because a "
                    f"browser's Origin header carries nothing after the port and an entry "
                    f"that carries more can never match one."
                )
                raise ValueError(msg)
        return value

    @property
    def is_loopback(self) -> bool:
        """Whether ``bind_host`` reaches only this machine.

        **Delegated to the bind policy rather than answered here**, because this was a second
        set and the two disagreed. This one held ``127.0.0.1``, ``::1`` and ``localhost``;
        :data:`manicule.app.bind.LOOPBACK_HOSTS` also holds ``::ffff:127.0.0.1`` and matches
        after stripping and lowercasing. So a configured ``::ffff:127.0.0.1`` was loopback to
        :func:`~manicule.app.bind.resolve_bind`, which would have bound it with no flag at all,
        and routable to the bind check that then lived in :meth:`Settings.policy_problems` —
        which raised out of ``build_container`` and refused *every* command over an address
        reaching nothing but this machine. That check has since moved to the two places a
        socket is actually made; delegating here is what keeps them agreeing.

        Imported inside the property because :mod:`manicule.app.bind` imports this module: the
        cycle is real at import time and gone by the time anything asks the question.
        """
        from manicule.app.bind import is_loopback  # noqa: PLC0415 - cycle: bind imports this

        return is_loopback(self.bind_host)


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
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)


class WebhookSettings(Section):
    url: str = Field(min_length=1)
    events: tuple[str, ...] = Field(min_length=1)
    secret: SecretStr | None = None
    retries: int = Field(default=2, ge=0, le=10)


class EventSettings(Section):
    transport: Literal["in_process", "webhook"] = "in_process"
    webhooks: tuple[WebhookSettings, ...] = ()


class QdrantSettings(Section):
    """How to reach the Qdrant server ``storage.vector_db`` selects, and what to keep on it.

    Where the server *is* lives in ``storage.vector_db_url``, beside ``storage.db_url``,
    because an endpoint is a property of the installation rather than of the client dialing
    it — and two places a location can be set is how the two come to disagree. The first five
    settings here are everything else the dial needs.

    The rest shape the collections: where the vectors and the payload are held, how the HNSW
    graph is built and when one is built at all, and whether a quantized copy of the vectors is
    searched. Three things are true of every one of them.

    **They are memory, recall and throughput dials, and never eligibility.** None of them
    changes which rows a filter admits. The graph and quantization settings move recall, so they
    can change which admitted chunks an approximate search ranks highest, but none changes which
    chunks are candidates at all — so none widens what ``docs/retrieval.md`` §3.3 holds level
    across the two backends, which is why they have no embedded-store counterpart, for the same
    reason ``storage.ann_index_threshold`` has no Qdrant one.

    **Each default is the value Qdrant gives a collection nobody tuned**, which is the
    collection every installation already has. An upgrade therefore changes nothing on a server
    running its stock configuration. A server whose own configuration chose other defaults has
    its existing collections brought to these values, because the collection is described here
    now rather than there.

    **They are applied to a collection that exists, not only to a new one.** Each time the
    store is prepared it reads the collection back and changes whatever differs, so an edit
    takes effect at the next start rather than waiting for a collection that is never created
    again (``docs/storage.md`` §6.7).

    The vector *datatype* is deliberately not among them. The integrity checksum is taken over
    the ``binary32`` values a point stores, and a ``float16`` or ``uint8`` collection hands back
    other numbers, so every row would read as corrupt and drop out of search; nor can a
    datatype be changed once a collection exists. Scalar quantization is the memory saving that
    keeps the checksummed originals.
    """

    api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key. Resolved from QDRANT_API_KEY when not set here, and "
        "held as a secret so that printing configuration cannot leak it.",
    )
    prefer_grpc: bool = Field(
        default=False,
        description="Speak gRPC on the data path instead of HTTP. Faster — a float32 vector "
        "crosses as bytes rather than as JSON decimal text — and it needs the gRPC port open "
        "and terminated, which an HTTP ingress in front of Qdrant usually does not do. HTTP "
        "is the default because it is the one that works through whatever is already there; "
        "neither transport changes a stored value.",
    )
    grpc_port: int = Field(
        default=6334,
        ge=1,
        le=65535,
        description="The gRPC port, used only when prefer_grpc is set. Qdrant's own default, "
        "and not carried by the URL: storage.vector_db_url names the HTTP endpoint.",
    )
    timeout_s: int = Field(
        default=30,
        ge=1,
        description="Wall clock for one request, in whole seconds — the unit the client "
        "takes. Its own default is five, which a bulk upsert over a slow link exceeds "
        "honestly rather than exceptionally.",
    )
    collection_prefix: str = Field(
        default=QDRANT_COLLECTION_PREFIX,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="What this installation's collections are called, and the only thing that "
        "keeps two installations sharing one server apart. A workspace digest and the "
        "fingerprint hash follow it, so one installation's workspaces and embedding spaces "
        "already cannot meet — but nothing identifies an installation on its own, and a "
        "workspace is called 'default' on both machines. Leave this at its default on a shared "
        "server and the two share collections. Give each its own.",
    )
    quantization: Literal["none", "scalar"] = Field(
        default="none",
        description="Keep an int8 copy of every vector, a quarter of its size, and pick search "
        "candidates with it. The float32 originals are kept beside it and score the candidates "
        "picked, so every score and checksum is still over them; which candidates are picked is "
        "the recall this trades. On its own it adds memory; paired with on_disk_vectors, it is "
        "what lets the originals leave RAM. Setting it back to 'none' drops the copy.",
    )
    quantization_always_ram: bool = Field(
        default=True,
        description="Hold the quantized copy in RAM even when on_disk_vectors puts the "
        "originals on disk. That pairing is what makes quantization a memory saving rather "
        "than a latency cost, because a search then reads originals only for the candidates it "
        "picked. Read only when quantization is 'scalar'.",
    )
    on_disk_vectors: bool = Field(
        default=False,
        description="Serve the float32 vectors from disk through the page cache rather than "
        "holding them in RAM. At 1024 dimensions each is four kilobytes, so on a large corpus "
        "this is most of the resident memory; pair it with scalar quantization to keep search "
        "off the disk.",
    )
    on_disk_payload: bool = Field(
        default=True,
        description="Keep each point's payload on disk rather than in RAM. The payload carries "
        "the whole chunk, so it is a second copy of the corpus text; the fields a query "
        "filters on are indexed and stay in RAM either way. Qdrant's own default, stated here "
        "so that it is not whatever a server's configuration happens to say.",
    )
    hnsw_m: int = Field(
        default=16,
        ge=4,
        description="Edges per node in the HNSW graph: more is better recall and a larger "
        "graph. Refused below four. Qdrant reads zero as no graph at all, and keeping search "
        "exhaustive already has one spelling, indexing_threshold_kb = 0.",
    )
    hnsw_ef_construct: int = Field(
        default=100,
        ge=4,
        description="Neighbors considered for each node while the graph is built: more is a "
        "better graph and a slower build, and costs nothing at query time. Qdrant refuses a "
        "value below four.",
    )
    indexing_threshold_kb: int = Field(
        default=10_000,
        ge=0,
        description="Kilobytes of vectors a segment holds before Qdrant builds an HNSW graph "
        "for it; below that the segment is searched exactly. Kilobytes rather than points: a "
        "1024-dimension vector is four, so the default indexes a segment past about 2,500 "
        "chunks. 0 never builds a graph. Unrelated to storage.ann_index_threshold, which "
        "counts vectors and is read only by the embedded store.",
    )


class StorageSettings(Section):
    """Where indexed data lives.

    One relational store, and two vector stores that are alternatives rather than layers.
    ``db`` is closed at ``sqlite``: the 35 modeled tables that own the corpus, durable
    acquisition, re-embedding, collections, versions and audit records have one implementation
    and naming another would advertise support that does not exist.

    ``vector_db`` is a real choice, and the two differ in where the index lives rather than in
    what it holds. ``lancedb`` is embedded — a directory under the data directory, no server,
    nothing to reach — and is the default because it is the configuration that works with
    nothing else running. ``qdrant`` puts the index on a server, which is what lets several
    processes share one and what lets a container keep no volume; it costs a network
    dependency, a backup procedure that is no longer a directory copy, and — because the chunk
    travels with the vector — a corpus that leaves this machine. Re-embedding's shadow
    generations are a LanceDB mechanism and stay one; a durable re-embed on any other backend
    is refused by name rather than half-performed.
    """

    db: Literal["sqlite"] = "sqlite"
    db_url: str | None = Field(
        default=None, description="Overrides the default path under the data directory."
    )
    vector_db: Literal["lancedb", "qdrant"] = "lancedb"
    vector_db_url: str | None = Field(
        default=None,
        description="Where the vector store is, for a backend that is somewhere. Required by "
        "'qdrant' — the Qdrant HTTP endpoint, e.g. https://qdrant.internal:6333 — and refused "
        "for 'lancedb', which lives under the data directory and would otherwise be reading a "
        "setting that looks like it is in force.",
    )
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    retain_source_bytes: bool = Field(
        default=True,
        description="Keep fetched bytes so re-parsing never means re-fetching. Turning this "
        "off makes every re-index a re-crawl.",
    )
    lifecycle_plan_schedule_s: float | None = Field(
        default=None,
        gt=0,
        description="Seconds between aggregate lifecycle dry runs in a served process. None "
        "disables scheduled planning; this setting never authorizes cleanup or deletion.",
    )
    source_history_retention_days: int | None = Field(
        default=None,
        ge=1,
        description="Optional age cutoff included in scheduled source-history dry runs.",
    )
    snapshot_plan_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional snapshot identity to include in scheduled deletion dry runs. "
        "The scheduler records aggregate impact only and cannot confirm deletion.",
    )
    ann_index_threshold: int = Field(
        default=100_000,
        ge=0,
        description="Vectors above which dense search stops being exhaustive and an IVF-PQ "
        "index is wanted. Below it search is exact and an index would trade recall for "
        "latency nobody is waiting on. The same number, applied to the rows an existing index "
        "does not cover, is when that index reads as stale. ``0`` keeps search exhaustive "
        "permanently and leaves any index already built alone. Read only by a vector store "
        "whose index manicule builds, which is 'lancedb'; 'qdrant' maintains its own and "
        "reports no lifecycle rather than reporting this one.",
    )

    checksum_backfill_batch: int = Field(
        default=512,
        ge=1,
        le=10_000,
        description="Vector rows one pass of `manicule vector-checksum --yes` reads and "
        "rewrites. The bound on both its memory and the work a crash can lose: the pass "
        "selects rows that record no checksum, so an interruption costs at most this many rows "
        "and the next pass resumes without a cursor. Larger is fewer commits — or, on a "
        "network-backed store, fewer round trips — over a big corpus; smaller is a shorter "
        "interval in which a rewrite is in flight.",
    )

    @field_validator("ann_index_threshold")
    @classmethod
    def _threshold_can_be_honored(cls, value: int) -> int:
        """Refuse a threshold no build could ever satisfy.

        An 8-bit product quantizer needs 256 vectors to train a codebook, and LanceDB refuses
        the build below that. A threshold of 50 therefore does not mean "index early" — it
        means every surface reports a build as due, forever, and every attempt to perform one
        is declined for a reason that has nothing to do with the number the operator set.
        Caught here, it is a typo at startup instead.

        Checked whatever the selected vector store is, and deliberately: the floor belongs to
        the setting rather than to a backend, so switching to a store that ignores the number
        does not quietly make an unusable value acceptable — and switching back would then fail
        at a moment that has nothing to do with the edit that caused it.

        Raises:
            ValueError: The threshold is neither ``0`` nor a value a build could reach.
        """
        if value != 0 and value < MINIMUM_ANN_INDEX_THRESHOLD:
            msg = (
                f"storage.ann_index_threshold is {value}, which no index can be built at: an "
                f"8-bit product quantizer needs {MINIMUM_ANN_INDEX_THRESHOLD} vectors to "
                f"train. Use 0 to keep search exhaustive, or a threshold of at least "
                f"{MINIMUM_ANN_INDEX_THRESHOLD}."
            )
            raise ValueError(msg)
        return value


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
    snapshot_promotion_policy: SnapshotPromotionPolicy = Field(
        default=SnapshotPromotionPolicy.REQUIRE_COMPLETE,
        description="Whether a source watermark requires a fully reproducible retained "
        "snapshot or may explicitly record typed omissions.",
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
    max_journal_records: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum durable discovery records admitted across unsettled runs. The "
        "writer reserves against this bound before acknowledging a source record, so an "
        "enumeration stops safely instead of turning a slow downstream stage into unbounded "
        "disk growth.",
    )
    max_journal_metadata_bytes: int = Field(
        default=1024 * 1024 * 1024,
        ge=1,
        description="Maximum encoded metadata bytes admitted across unsettled discovery "
        "records. Separate from the record count because opaque fetch references and source "
        "metadata vary in size by orders of magnitude.",
    )
    max_acquired_blob_backlog_bytes: int = Field(
        default=20 * 1024 * 1024 * 1024,
        ge=1,
        description="Maximum retained source bytes waiting for offline indexing. Admission "
        "is reservation-based so concurrent acquisitions cannot each pass the same stale "
        "capacity check.",
    )
    min_disk_headroom_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1,
        description="Free filesystem space preserved while admitting journal metadata and "
        "acquired blobs. A write that would cross this floor is refused before its durable "
        "record is acknowledged.",
    )
    target_batch_tokens: int = Field(
        default=65_536,
        ge=1,
        description="Tokens per embedding coordinator batch. The batch *size* is derived from "
        "this and the chunk budget; 64K yields 64 coordinator rows for 1,024-token chunks. "
        "The embedder can split those rows into smaller forward passes. Lower it when the "
        "coordinator is memory-bound.",
    )
    max_embed_batch: int = Field(default=64, ge=1, description="Upper clamp on the derived size.")
    reembed_validation_page: int = Field(
        default=1024,
        ge=1,
        description="Physical shadow rows checked per validation page. Larger pages reduce Lance "
        "reader overhead; lower this when validation memory is constrained.",
    )
    rebuild_replay_page: int = Field(
        default=256,
        ge=1,
        description="Staged documents per replay checkpoint. Vector copies remain independently "
        "bounded by bytes and 512 rows; lower this only when retained replacement payloads are "
        "memory-bound.",
    )
    rebuild_validation_page: int = Field(
        default=100,
        ge=1,
        description="Staged documents per rebuild validation checkpoint. Vector checks remain "
        "independently bounded to 512 rows; lower this only when retained replacement payloads "
        "are memory-bound.",
    )
    queue_depth_factor: int = Field(
        default=2,
        ge=1,
        description="Bounded queue depth, as a multiple of the consumer's parallelism. "
        "With durable acquisition wired, this bounds how far the journal reader and fetch "
        "workers may run ahead of local parsing and embedding; the legacy fallback applies "
        "the same bound directly to discovery.",
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
        default="onnx",
        min_length=1,
        description="Which embedder implementation to use. ``onnx`` is the one manicule ships "
        "and runs everywhere. ``mlx`` is Metal-native and roughly four to five times faster on "
        "Apple Silicon, and it is a separate install — ``manicule-mlx``, which is "
        "GPL-3.0-or-later where manicule is MIT. Switching between them never re-embeds: "
        "``backend`` is excluded from the embedding fingerprint's identity, and the two agree "
        "to cosine 0.99999998. Both embed **in this process**, and ``onnx`` runs on any CPU — "
        "without AVX2 it takes onnxruntime's slower kernels rather than refusing. ``ollama`` "
        "is the third option and a different kind: the model runs on a server, which is worth "
        "it where in-process embedding is too slow to be practical and not otherwise. It is a "
        "separate install — ``manicule-ollama`` in ``packages/`` — and a separate vector "
        "space, because nothing has measured a served model against these two.",
    )
    model: str = Field(default="BAAI/bge-m3", min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    prefix_scheme: PrefixScheme = Field(
        default=PrefixScheme.NONE,
        description="The asymmetric query/document prefixes this model was trained with. "
        "``none`` — the default, and right for ``BAAI/bge-m3``, which was trained without "
        "any. ``nomic`` is ``search_document: ``/``search_query: `` and is what makes "
        "``nomic-embed-text`` usable; ``qwen3`` is Qwen3-Embedding's query-side instruction. "
        "It sits here rather than under a backend because both sides have to move together "
        "and a backend cannot tell which side it is serving — so it is chosen once, applied "
        "at ingest and at query, and recorded in the embedding fingerprint. Changing it "
        "costs a full re-embed, for the same reason changing the model does: every stored "
        "vector was computed from different text.",
    )
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
        description="Which **vendor or authenticated local CLI** serves the model: ``ollama``, "
        "``openai``, ``anthropic``, ``codex`` or ``claude``. This is what decides which "
        "credential is needed and whether the endpoint leaves this machine — not which "
        "component is built. Select generator ``cli`` for the Codex and Claude commands.",
    )
    generator: str = Field(
        default="litellm",
        min_length=1,
        description="Which registered generator **component** to build. Separate from "
        "``provider`` because the two answer different questions and conflating them made "
        "the default configuration unrunnable: one implementation reaches every vendor "
        "through a base_url, so the component is not a function of the vendor. Use ``cli`` "
        "to ask through an installed Codex or Claude command; other values select "
        "third-party generators.",
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


class GlossarySettings(Section):
    """Glossary-aware entity and acronym retrieval (``docs/retrieval.md`` §14).

    Two settings decide whether a query is expanded and one decides whether an already-indexed
    definition is trusted enough to expand it. There is deliberately **no** setting that
    resolves a conflict: two definitions of one term in scope are reported as a conflict
    whatever this section says, because a configurable tie-break is a silent choice with an
    audit trail rather than an absence of one.
    """

    enabled: bool = Field(
        default=True,
        description="Whether a query naming a glossary term is expanded and its definition "
        "promoted. Off means no lookup runs at all, rather than a lookup whose answer is "
        "discarded: a disabled feature that still queries a store can still be slow and can "
        "still fail.",
    )
    detect_on_ingest: bool = Field(
        default=True,
        description="Whether ingest reads definitions out of documents. Separate from "
        "``enabled`` because the two fail differently: turning this off stops new entries "
        "being written and leaves existing ones queryable, which is what an operator wants "
        "while investigating a detector that is producing rubbish.",
    )
    min_entry_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Detection confidence an entry needs before a query will act on it. "
        "Applied at query time as well as at ingest, so raising it takes effect against a "
        "corpus already indexed — the only remedy available to someone who cannot re-ingest.",
    )
    max_terms: int = Field(
        default=3,
        ge=1,
        description="Distinct terms one query may expand. A query naming four of them is "
        "asking something a definition lookup cannot answer, and expanding all four produces "
        "a second query that is a list of noun phrases.",
    )
    homographs: tuple[str, ...] = Field(
        default=(),
        description="Extra terms to treat as ordinary English words, so they expand only on an "
        "exact-case match or a definitional question. Extends the shipped list rather than "
        "replacing it: the words that collide with a corpus's terms are a property of that "
        "corpus, and nobody should have to restate the common ones to add one of their own.",
    )


class ResearchSettings(Section):
    """Multi-step research: what one run may spend, and what its report may read.

    The fields are ceilings rather than targets. A loop whose only bound is "until the model
    says stop" hands an unattended caller the machine for as long as a model feels like
    planning — the argument ``docs/surfaces.md`` §4 already made about corpus-scale operations,
    and the reason a *bounded* one may be reached from a surface an assistant calls.

    ``report_tokens`` is deliberately wider than any profile's ``context_tokens``, so it is
    **not** covered by the profile's startup window cross-check and gets its own. See
    :func:`manicule.research.loop.plan_problem`.
    """

    max_cycles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Rounds of planning and searching. Each round after the first spends a "
        "model call deciding whether it is worth running, so this bounds latency as much as "
        "retrieval.",
    )
    max_sub_questions: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Searches one cycle may run. Times ``max_cycles`` this is the ceiling on "
        "retrievals for one question, and retrievals are the expensive part: each is an "
        "embedder pass and, on a reranking profile, a cross-encoder pass.",
    )
    concurrency: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Retrievals in flight at once. Small deliberately: the embedder serializes "
        "every forward pass through one worker thread, so a wider fan-out queues there while "
        "each task still holds a database connection, and the pool is what runs out.",
    )
    report_passages: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Passages the report may cite from. The reason a research report is worth "
        "more than one ask: a question with four facets cannot be answered out of the five "
        "passages a profile's ``final_top_k`` allows.",
    )
    report_tokens: int = Field(
        default=12288,
        ge=1024,
        description="Context budget for the report, in the generator's tokenizer. Checked "
        "against the served context window before the first question rather than discovered "
        "by a server truncating the prompt.",
    )
    timeout_s: float = Field(
        default=300.0,
        gt=0,
        description="Wall clock for one run, checked between cycles. The loop stops planning "
        "further cycles once it is reached and reports with what it has: a run that returns "
        "late is better than one that returns nothing.",
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
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)
    assert_scope: bool = Field(
        default=False,
        description="Run the pipeline's scope assertion on every query, as a runtime check "
        "rather than only in the suite. Off by default because it costs a document lookup per "
        "candidate per stage; on, it holds a live pipeline to the property that makes the "
        "vector store's ``workspace_ids`` exemption safe.",
    )
    cross_workspace_limit: int = Field(
        default=8,
        ge=2,
        le=64,
        description="The most workspaces one administrator's search may span. Each is its own "
        "scoped query, so this bounds the work one request can ask for.",
    )


class ConnectorSettings(Section):
    """One configured source.

    **``schedule_s`` is back, and what changed is that there is now a scheduler.** #98 deleted
    the field because it configured something that did not exist: it was reported by
    ``connector list`` and read by nothing, so it was a promise the software did not keep and
    would have been cited as evidence that scheduling worked. It was removed rather than
    documented as unimplemented, which is why a configuration carrying it has been refused
    loudly ever since instead of silently ignored.

    :class:`~manicule.app.served.Scheduler` is what makes it true. A served manicule reads this
    field, and nothing else does — an unserved installation has no process to run a schedule in,
    which is the same reason write commands need a server at all.
    """

    type: str = Field(min_length=1, description="Registered connector implementation.")
    enabled: bool = Field(
        default=True,
        description="Whether `manicule connector sync` will run this source. False refuses the "
        "sync rather than skipping it quietly, so a disabled source that somebody tries to run "
        "says so.",
    )
    schedule_s: float | None = Field(
        default=None,
        gt=0,
        description="Seconds between automatic syncs of this source, run by `manicule serve`. "
        "None is the default and means this source syncs only when somebody asks for it.",
    )
    """How often a served manicule syncs this source on its own.

    **Per source rather than one global interval**, because the sources in one installation are
    not alike: a handbook that changes weekly and a runbook directory that changes hourly want
    different answers, and a single number would be tuned for whichever one somebody noticed.

    **``enabled = false`` is honored here too**, and it has to be: a schedule is exactly the
    place a disabled source would come back to life without anybody typing anything. A disabled
    source is never scheduled, whatever this says.

    **The first sync happens after one interval, not at startup.** A server that swept every
    scheduled source the moment it started would turn a restart — which is how a session is
    re-taken, and therefore something an operator does deliberately and often — into a full
    corpus sync nobody asked for.
    """
    retain_source_bytes: bool | None = Field(
        default=None,
        description="Override storage.retain_source_bytes for this connector. None inherits "
        "the installation-wide default.",
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict, description="Validated against the connector's own config model."
    )


class BrowserProvider(StrEnum):
    """How `connector login` gets hold of a Confluence session.

    Four ways in, and naming them makes the choice a workspace's rather than a flag's. The two
    browser members are deliberately *not* one member with a boolean beside it: which browser
    opens is the whole question on a desktop whose identity provider treats an unfamiliar one
    differently, and a setting that could not distinguish them would be a setting that could not
    express the case this exists for.
    """

    INSTALLED_CHROMIUM = "installed_chromium"
    """A supported Chromium-family browser already installed on this machine, driven through a
    dedicated manicule profile. What an identity provider is most likely to recognize."""

    BUNDLED_CHROMIUM = "bundled_chromium"
    """The Chromium Playwright downloads. Needs nothing installed and is the most likely to be
    refused by a conditional-access policy, because it is a device the policy has never seen."""

    BROWSER_STATE = "browser_state"
    """Import a Playwright `storage_state` document somebody else's tooling wrote."""

    MANUAL_COOKIE = "manual_cookie"
    """Paste a `Cookie` header from a browser you signed in to yourself.

    The only member that drives no browser, and therefore the only one under which manicule
    *cannot* see the page a password is typed into rather than merely not looking. It is the
    historical default and it is not deprecated — see `manicule.connectors.sessions`.
    """


class ConfluenceAuthSettings(Section):
    """How this workspace authenticates to Confluence, and what it keeps afterwards.

    **Workspace-scoped rather than per connector, which is a departure worth its own paragraph.**
    Every other Confluence setting lives in `[connectors.<name>].options` and is validated as
    `ConfluenceConfig`, because every other Confluence setting is a fact about a *source*. These
    are not. Which browser is installed is a fact about this machine, and a held session is
    keyed by authority rather than by connector — so two connectors pointed at one wiki share one
    sign-in, and per-connector provider settings would let them disagree about a decision only
    one of them can make. The setting lives where the thing it configures lives.

    **Nothing here writes a session to disk, and there is no setting that would.** A captured
    session crosses to the server over the control socket and lives in its memory until the
    process ends — the property `manicule.connectors.sessions` is built around. A restart means
    signing in again, which against a session whose own lifetime is `session_max_age_hours` is a
    small cost for never having a credential at rest.

    **There is no `fallback` dial either, and its absence is the design.** A failed provider
    raises a typed refusal naming the alternatives; it never quietly becomes a different provider. A
    single-valued setting saying so would configure nothing, and this repository has already
    removed one of those (`schedule_s`, #98) rather than ship a promise the code did not keep.
    """

    default_provider: BrowserProvider = Field(
        default=BrowserProvider.MANUAL_COOKIE,
        description="What `manicule connector login <name>` uses when no provider flag is "
        "given. The default is the paste prompt, which is what every workspace written before "
        "this setting existed already does.",
    )
    """The provider a bare `connector login` selects.

    Defaulted to `manual_cookie` rather than to a browser, so that adding this section to the
    settings tree changes nothing for an installation that does not set it. A default that
    opened a browser would make an upgrade the moment somebody's `connector login` started
    launching a window they did not ask for.
    """

    installed_browser: str = Field(
        default="",
        description="Which installed browser `installed_chromium` drives: a supported name "
        "(`chrome`, `chromium`, `edge`, `brave`) or an absolute path to an executable. Empty "
        "discovers one, and refuses rather than guessing when several are present.",
    )

    profile_dir: Path | None = Field(
        default=None,
        description="The dedicated profile directory `installed_chromium` signs in under. "
        "Never your ordinary browser profile. Unset uses a private directory beneath the data "
        "directory.",
    )
    """Where the authentication profile lives.

    **A separate profile rather than the person's own, and this is a security boundary rather
    than tidiness.** An ordinary daily-use profile is not available to an unrelated process by
    design, and the ways to take it anyway — remote debugging on a running browser, copying the
    profile, decrypting the cookie database — are all things manicule refuses to do. What it can
    do honestly is sign in to a profile of its own, which is why this exists.

    It holds live session cookies once used, so it is created user-only and documented as
    changing the at-rest security boundary of the installation.
    """

    @field_validator("profile_dir")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        """Expand `~`, as the data and cache directories already do.

        Without this a configured `~/.local/share/...` is a directory literally named `~` in
        whatever the process's working directory happened to be — which for a launchd-started
        server is not anywhere the operator would think to look, and which would hold a live
        browser profile.
        """
        return None if value is None else value.expanduser()


class AuthenticationSettings(Section):
    """Per-protocol interactive authentication. Confluence is the only one that has any."""

    confluence: ConfluenceAuthSettings = Field(default_factory=ConfluenceAuthSettings)


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


class AuthoringSettings(Section):
    """Where an authored document is allowed to land, and in which collections.

    ``document_create`` writes a markdown file into a configured filesystem source's root and
    then indexes that path, so the file stays the record and the connector stays how content
    enters. What this section decides is the *bounds* of that authority, and it is two names
    rather than one because they answer two questions an operator would not want answered by
    the same setting: which directory may be written into, and which groupings within it a
    caller may add to.

    **Both are empty by default, and empty means authoring is off.** The operation exists on
    every surface whether or not this is set; unconfigured, it refuses and names the settings.
    That is the safe direction for a write, and it is the reason no surface needs a switch of
    its own: an installation that never configured authoring has none, including over a socket.

    The corpus this was built for is read as *instructions* rather than as data — assistants
    treat recalled guidance as standing direction — so write access here is the ability to
    place text in front of future sessions. Bounded authority is the whole design: one
    workspace, one configured source, one of a named set of collections, at a path the caller
    never supplies.
    """

    source: str = Field(
        default="",
        description="A configured connector *instance* name — the key in "
        "``[connectors.<name>]`` — which must be a filesystem source. Its root is the only "
        "directory an authored document can be written beneath, and its name becomes the "
        "``source`` half of the document's identity, so a document authored here and the same "
        "file later re-synced by that connector are one document rather than two.",
    )
    collections: tuple[str, ...] = Field(
        default=(),
        description="The collections a caller may author into, by name. A collection this "
        "does not list is refused even when it exists in the workspace: the point of naming "
        "them is that adding a collection to manicule does not silently widen what may be "
        "written. Each name is also the directory beneath the root that its documents land "
        "in, so it must be a single path segment — and the collection wants a "
        "``CollectionRule`` prefix saying so, or only the documents `document_create` wrote "
        "are in it. `manicule doctor` reports a collection named here that has neither.",
    )

    @property
    def configured(self) -> bool:
        """Whether authoring has been switched on by naming both a source and a collection.

        Both, because either alone describes an operation that cannot run: a source with no
        collection has nowhere to put a document that is not an unscoped pile, and collections
        with no source have no root to be written beneath. Reporting "configured" for half of
        it would move the refusal from the setting to the first call.
        """
        return bool(self.source and self.collections)


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
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    connectors: dict[str, ConnectorSettings] = Field(default_factory=dict)
    authentication: AuthenticationSettings = Field(default_factory=AuthenticationSettings)
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
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    authoring: AuthoringSettings = Field(default_factory=AuthoringSettings)

    @classmethod
    @override
    # `settings_customise_sources` is pydantic-settings' own name for the hook it calls, and a
    # name defined outside this repository is spelled the way its definition spells it. Under
    # any other spelling this overrides nothing and the source order silently reverts.
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
        """Fill provider and vector-store credentials from the environment by convention."""
        del context
        environment = provider_environment()
        self.providers = resolve_provider_keys(
            self.providers, self.selected_providers, environ=environment
        )
        if self.storage.qdrant.api_key is None:
            supplied = environment.get(QDRANT_API_KEY_ENV, "").strip()
            if supplied:
                self.storage.qdrant.api_key = SecretStr(supplied)

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
        object instead would hand out every API key, OAuth client secret, session signing key
        and webhook signing key to anyone allowed to read configuration.

        Masking *credentials*, and unrelated to :class:`RedactionSettings`, which removes
        personal data from text sent to a model. Same word, two features.
        """
        dumped: Any = self.model_dump(mode="json")
        return _mask(dumped, type(self))

    # --- policy -------------------------------------------------------------------------

    def policy_problems(self) -> list[str]:
        """Configurations that are individually valid and jointly wrong.

        Checked once at startup, before anything is constructed, so an impossible setup fails
        immediately instead of at the first request that happens to exercise it.

        **Every problem here is a problem for any command**, and that is the rule for what
        belongs. A wide ``security.transport.bind_host`` with ``security.auth.mode`` at ``none``
        used to be listed, and it is not a problem for `manicule index --stats`: that command
        opens no socket, so refusing it enforced a rule about *listening* against every process
        that merely reads the data directory. In a container whose server runs with
        ``--no-authentication`` — the deployment the flag exists for — it meant
        ``kubectl exec … manicule doctor`` could not run, which is precisely the tool an operator
        reaches for when something is wrong with that deployment.

        That rule is enforced where a socket actually comes into being, and nowhere else:
        :func:`~manicule.app.bind.resolve_bind` before an address exists, and
        :func:`~manicule.api.app._require_auth_for_wide_bind` before an application does. Both
        refuse, both take ``--no-authentication`` to waive, and ``doctor``'s ``transport`` check
        reports the same condition as a finding for anybody who wants to know without serving.

        ``security.auth.mode = 'oauth'`` with no provider left for the same reason: it stops a
        browser from signing in, not ``manicule index`` from running.
        :func:`~manicule.app.people.serving_problems` holds it with the rest of what a sign-in
        needs, ``build_app`` refuses to serve with any of it, and ``doctor``'s ``sign_in`` check
        reports it.
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
            cli_owns_auth = (
                self.llm.generator.strip().lower() == "cli"
                and self.llm.provider.strip().lower() == name
                and self.embedding.provider.strip().lower() != name
                and name in CLI_AUTH_PROVIDERS
            )
            if needs_credential(name) and not (cli_owns_auth or self.provider(name).has_key):
                expected = " or ".join(env_var_names(name))
                cli_hint = (
                    " To use its existing local CLI login instead, set llm.generator to 'cli'."
                    if name in CLI_AUTH_PROVIDERS
                    else ""
                )
                problems.append(
                    f"provider {name!r} is selected but has no API key. Set {expected}, or "
                    f"providers.{name}.api_key.{cli_hint}"
                )

        problems.extend(self._redaction_problems())
        problems.extend(self._source_restriction_problems())
        problems.extend(self._vector_store_problems())
        problems.extend(self._dispatch_problems())

        if self.security.audit.destination is AuditDestination.WEBHOOK and not self.events.webhooks:
            problems.append("security.audit.destination is 'webhook' but events.webhooks is empty")

        return problems

    def _dispatch_problems(self) -> list[str]:
        """Settings that name a delivery this build does not perform.

        ``events.transport``, ``events.webhooks`` and ``security.audit.destination`` all parse,
        all validate against each other, and all reach nothing: there is no event bus and no
        webhook dispatcher — they are the unchecked half of
        `#14 <https://github.com/mgd43b/manicule/issues/14>`_. An operator who configures an
        endpoint for audit events and sees the process start has been told the events are going
        somewhere.

        This is the rule ``CONTRIBUTING.md`` states rather than a new one: *a setting that
        appears to be in force and silently is not is worse than one that fails at startup*. So
        it fails at startup, and says what is missing rather than that the value is wrong —
        because the value is not wrong, it is early.
        """
        problems: list[str] = []
        if self.events.transport != "in_process" or self.events.webhooks:
            problems.append(
                "events.transport or events.webhooks names a webhook, and nothing in this "
                "build dispatches one — there is no event bus yet (#14). Leave "
                "events.transport at 'in_process' and remove events.webhooks until there is."
            )
        if self.security.audit.destination is not AuditDestination.LOCAL:
            problems.append(
                f"security.audit.destination is "
                f"{self.security.audit.destination.value!r}, and this build writes the audit "
                f"trail locally and nowhere else (#14). Set it to 'local', or the records you "
                f"are expecting elsewhere are not being sent."
            )
        return problems

    def _vector_store_problems(self) -> list[str]:
        """Vector-store settings that cannot do what they say, and the egress one.

        Three refusals, and the third is the one this method exists for.

        **An endpoint the selected backend does not dial**, or does not have. ``lancedb`` lives
        under the data directory and reads no URL; ``qdrant`` is nothing without one. Either
        mismatch leaves a setting that appears to be in force and is not.

        **A corpus that leaves the machine while the policy says it may not.** The chunk travels
        with the vector (``docs/storage.md`` §6.2), so a vector store on another host is an
        egress path for document *text*, not merely for embeddings — and it is one
        :attr:`selected_endpoints` cannot see, because that records model endpoints and a
        database is not one. Without this check a local-only data policy would report itself
        satisfied while every ingest wrote the corpus to another machine, which is worse than
        having no policy at all.

        **A ``local_only`` source indexed into a remote store.** ``local_only`` is a floor that
        no exemption releases (``docs/generation.md`` §7.5), and the argument it rests on is
        that search never leaves this machine. A network-backed vector store falsifies exactly
        that premise for every source, so the two are refused together rather than the floor
        being quietly lowered for the sources that most depend on it.
        """
        storage = self.storage
        problems: list[str] = []
        url = (storage.vector_db_url or "").strip()

        if storage.vector_db == "qdrant" and not url:
            problems.append(
                "storage.vector_db is 'qdrant' but storage.vector_db_url is empty. A "
                "network-backed vector store has nowhere to be by default; set it to the "
                "Qdrant HTTP endpoint, e.g. https://qdrant.internal:6333."
            )
        if storage.vector_db == "lancedb" and url:
            problems.append(
                f"storage.vector_db_url is {url!r} but storage.vector_db is 'lancedb', which "
                f"lives in a directory under the data directory and dials nothing. That "
                f"setting is not in force; remove it, or select a served vector store."
            )
        if storage.vector_db != "qdrant" or not url:
            return problems

        if endpoint_egress(url).leaves_machine:
            policy = self.security.data_policy
            if not policy.cloud_allowed:
                problems.append(
                    f"security.data_policy.cloud_allowed is false, but storage.vector_db_url "
                    f"is {url!r}, which is not on this machine. The chunk's text is stored "
                    f"beside its vector, so every ingest would write the corpus to another "
                    f"host. Point it at loopback, choose the 'lancedb' vector store, or allow "
                    f"cloud processing."
                )
            restricted = policy.source_restrictions.local_only
            if restricted:
                named = ", ".join(sorted(restricted))
                problems.append(
                    f"security.data_policy.source_restrictions.local_only names {named}, but "
                    f"storage.vector_db_url is {url!r}, which is not on this machine. "
                    f"local_only is a floor that rests on search staying local, and indexing "
                    f"into a remote vector store sends those documents' text off this machine "
                    f"before any question is asked. Use the 'lancedb' vector store for a "
                    f"corpus with local-only sources, or stop restricting them."
                )
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

REDACTED: Final = "**********"
"""What a credential is displayed as. One constant, so the masking and the tests agree."""


def looks_secret(key: str) -> bool:
    """Whether a field *name* identifies a credential.

    **The fallback, not the rule.** Secrecy is decided from the declared type wherever there is
    one — see :func:`secret_setting` — because a name is a bad proxy for it in both directions,
    and this codebase had both:

    * ``llm.first_token_timeout_s``, ``llm.token_safety_factor``, ``llm.token_drift_tolerance``,
      ``rag.context.system_prompt_tokens`` and ``ingest.target_batch_tokens`` are floats and
      ints that contain "token". Masked, and refused by ``config set``, so five ordinary
      settings could not be inspected or changed. The exception list below grew out of exactly
      this, one name at a time, and could only ever cover the names somebody had already hit.
    * ``security.data_policy.auto_redact.hash_salt`` is a real ``SecretStr`` and matches none of
      the markers, so the one thing this function exists to hide was printed in the clear.

    What is left for it is the subtree that has no declared type at all: ``plugins.config``
    holds arbitrary per-component options, validated against each component's own model rather
    than against :class:`Settings`, so a name is genuinely all there is to go on there.
    """
    normalized = key.replace("-", "").replace("_", "").lower()
    if normalized in {"maxtokens", "overlaptokens"} or "tokenizer" in normalized:
        # Counts and tokenizer identities are public policy, not bearer tokens. Kept because
        # `plugins.config` still reaches this, and a component option called `max_tokens` is at
        # least as likely there as it was here.
        return False
    return any(marker.replace("_", "") in normalized for marker in _SECRET_KEYS)


def _declares_secret(annotation: object) -> bool:
    """Whether a field's annotation is ``SecretStr``, including ``SecretStr | None``."""
    return annotation is SecretStr or SecretStr in get_args(annotation)


def _field_model(annotation: object) -> tuple[type[BaseModel] | None, bool]:
    """The model behind a field, and whether it sits under arbitrary keys.

    ``keyed`` is True for ``dict[str, SomeModel]`` — ``llm.providers``, the connector table —
    where the next path segment is a name somebody chose rather than a declared field.
    """
    candidates = (annotation, *get_args(annotation))
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate, False
    for candidate in candidates:
        if get_origin(candidate) is not dict:
            continue
        for value in get_args(candidate):
            if isinstance(value, type) and issubclass(value, BaseModel):
                return value, True
    return None, False


def secret_setting(parts: Sequence[str]) -> bool:
    """Whether a dotted configuration key names a credential.

    Resolved against :class:`Settings` rather than guessed from the last segment, so a float
    called ``token_safety_factor`` is settable and a ``SecretStr`` called ``hash_salt`` is not.
    Falls back to :func:`looks_secret` the moment the path leaves the declared model, which is
    what ``plugins.config`` does immediately.
    """
    model: type[BaseModel] | None = Settings
    index = 0
    while index < len(parts):
        if model is None:
            return looks_secret(parts[-1])
        field = model.model_fields.get(parts[index])
        if field is None:
            return looks_secret(parts[-1])
        if index == len(parts) - 1:
            return _declares_secret(field.annotation)
        inner, keyed = _field_model(field.annotation)
        # A keyed table spends the next segment on the name, not on a field.
        index += 2 if keyed else 1
        model = inner
    return False


def _mask(
    value: Any,  # noqa: ANN401 - recursive over decoded JSON
    model: type[BaseModel] | None = None,
    key: str = "",
) -> Any:  # noqa: ANN401 - recursive over decoded JSON
    """Replace every declared credential with :data:`REDACTED`, walking the model alongside.

    The model is carried so secrecy is a fact about the *field* rather than about its name. It
    goes ``None`` as soon as the walk leaves :class:`Settings` — inside ``plugins.config``,
    whose contents are validated per component — and from there :func:`looks_secret` decides,
    which is the best available answer for a subtree with no declared type.

    **A dict with no model is that untyped subtree**, and it is judged by name. An earlier
    version of this function recursed into one without judging anything, so a plugin's
    ``api_key`` came back from ``config show`` in the clear while :func:`_strip` — which reaches
    the same conclusion through :func:`secret_setting` — correctly omitted it from the file.
    That is precisely the disagreement between what is hidden on display and what is withheld on
    disk that one predicate exists to prevent.

    The distinction the model alone cannot carry is that ``model is None`` means two different
    things: a *declared* field whose annotation is a scalar, already settled by
    :func:`_declares_secret` at its parent, and an *undeclared* subtree with nothing to settle
    it. Only a dict reaches here in the second case, because a declared scalar is returned by
    its parent and never recursed into.
    """
    if isinstance(value, dict):
        entries = cast("dict[str, Any]", value)
        if model is None:
            return {name: _untyped(item, name) for name, item in entries.items()}
        masked: dict[str, Any] = {}
        for name, item in entries.items():
            field = model.model_fields.get(name)
            if field is None:
                # A key the model does not declare. Same subtree, same rule.
                masked[name] = _untyped(item, name)
                continue
            if item is not None and _declares_secret(field.annotation):
                masked[name] = REDACTED
                continue
            inner, keyed = _field_model(field.annotation)
            if keyed and isinstance(item, dict):
                under = cast("dict[str, Any]", item)
                masked[name] = {key_: _mask(value_, inner, key_) for key_, value_ in under.items()}
            else:
                masked[name] = _mask(item, inner, name)
        return masked
    if isinstance(value, list):
        return [_mask(item, model, key) for item in cast("list[Any]", value)]
    return value


def _untyped(value: Any, key: str) -> Any:  # noqa: ANN401 - recursive over decoded JSON
    """Mask one entry of a subtree :class:`Settings` does not describe, by its name.

    The fallback the module docstring promises and :func:`secret_setting` already applies on the
    writing side, so ``config show`` and ``save_settings`` agree about a plugin's credentials.
    """
    if value is not None and looks_secret(key):
        return REDACTED
    return _mask(value, None, key)


__all__ = [
    "APP_NAME",
    "ENV_PREFIX",
    "REDACTED",
    "AtRestSettings",
    "AuditDestination",
    "AuditSettings",
    "AuthMode",
    "AuthSettings",
    "AuthoringSettings",
    "ConnectorSettings",
    "ContextSettings",
    "DataPolicySettings",
    "EmbeddingSettings",
    "EventSettings",
    "GlossarySettings",
    "IngestSettings",
    "LlmSettings",
    "LoggingSettings",
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
    "secret_setting",
]
