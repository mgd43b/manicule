"""What registration needs to know about the Confluence connector, without importing it.

The same split the parsers make (:mod:`manicule.parsers.config`): registration needs the
configuration model eagerly, so settings written for the connector are validated rather than
silently ignored, and nothing else. This module imports nothing heavier than pydantic, so
plugin discovery — which runs in every process that starts, before any configuration is
read — does not load an HTTP client on a machine that is never going to sync anything.
:mod:`manicule.connectors.cql` is imported for one predicate, whether a configured root page id
is a content id, and it is pure text and datetime work with no dependency of its own.

**The index this connector builds is not permission-aware.** Every page and attachment is
fetched as the account whose credentials are configured here, so the index ends up holding
everything that account can see, and anyone who can search this manicule installation can
retrieve any of it. Confluence's own space and page restrictions do not travel with the
content. That is a real widening of who can read what, it is a consequence of the design
rather than a defect in it, and it is stated here as well as in
``docs/connectors/confluence.md`` §9 because a configuration file is where somebody decides
which spaces to point it at.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from manicule.connectors import globs
from manicule.connectors.cql import is_page_id
from manicule.connectors.enriched import DEFAULT_PROFILE, EnrichedProfile
from manicule.connectors.pagination import origin_of
from manicule.connectors.site_routes import normalize_base_url, normalize_repository_path
from manicule.core.errors import ConfigError

__all__ = [
    "ALWAYS_EXCLUDED_LABELS",
    "CONNECTOR_NAME",
    "FILESYSTEM_CONNECTOR_NAME",
    "GIT_SITE_CONNECTOR_NAME",
    "GMAIL_CONNECTOR_NAME",
    "MBOX_CONNECTOR_NAME",
    "MBOX_REPRESENTATION_VERSION",
    "REPRESENTATION_VERSION",
    "SNAPSHOT_CONNECTOR_NAME",
    "AuthMethod",
    "ConfluenceConfig",
    "ConfluenceSnapshotConfig",
    "Deployment",
    "EnrichedProfile",
    "FilesystemConfig",
    "FullInventoryAuthority",
    "GitSiteConfig",
    "GmailConfig",
    "MboxConfig",
    "resolve_credentials",
    "resolve_gmail_client_secret",
    "resolve_gmail_credentials",
    "resolve_label_ids",
]

CONNECTOR_NAME = "confluence"
"""The registered name of the connector, and what configuration selects it by."""

FILESYSTEM_CONNECTOR_NAME = "filesystem"
"""The registered name of the local-directory connector."""

MBOX_CONNECTOR_NAME = "mbox"
"""The registered name of the mbox-archive connector."""

MBOX_REPRESENTATION_VERSION = "mbox/1"
"""What the mbox connector delivers a message as, and every message's version token.

A message's identity is the digest of its bytes (``docs/connectors/mbox.md`` §4), so a stored
message can never have changed and one constant stands for all of them. It is part of the scope
fingerprint, so changing what is delivered discards the watermark and the next sync reads every
file again.
"""

SNAPSHOT_CONNECTOR_NAME = "confluence-snapshot"
"""The registered name of the offline Confluence-snapshot connector.

A name of its own rather than a mode of ``confluence``, because the two share no
configuration: this one has no base URL, no credential and no deployment, and folding them
together would mean a config model where over half the fields are refused depending on
another field's value. It also keeps the credential refusal honest — a connector that
reaches no network cannot be misconfigured into trying.
"""

GIT_SITE_CONNECTOR_NAME = "git-site"
"""The registered name of the commit-pinned local website connector."""

GMAIL_CONNECTOR_NAME = "gmail"
"""The registered name of the Gmail connector."""

REPRESENTATION_VERSION = "rfc822/1"
"""What the Gmail connector fetches and hands to the parser, as a version token.

Part of the scope fingerprint (:meth:`GmailConfig.scope_fingerprint`), because a stored message
is never fetched again by id: changing what is fetched has to discard the stored history
position, or no message would ever be re-fetched (``docs/connectors/gmail.md`` §4.1). Bump the
number when the representation changes.
"""

ALWAYS_EXCLUDED_LABELS = ("SPAM", "TRASH", "DRAFT")
"""The labels no Gmail configuration can bring back into scope (``gmail.md`` §3)."""

GMAIL_QUOTA_UNITS_PER_MINUTE = 6_000
"""Gmail's per-user limit, in quota units a minute. The ceiling for ``quota_units_per_minute``."""

GMAIL_RAW_MESSAGE_UNITS = 60
"""What one ``messages.get?format=raw`` costs. A budget below it could never fetch a message."""


class GitSiteConfig(BaseModel):
    """One public website whose page inputs live in a local Git repository."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1, max_length=4_096)
    revision: str = Field(default="HEAD", min_length=1, max_length=1_024)
    content_root: str = Field(default=".", min_length=1, max_length=4_096)
    base_url: str = Field(min_length=1, max_length=4_096)
    include: tuple[str, ...] = Field(
        default=("**/*.md", "**/*.mdx", "**/*.html"), min_length=1, max_length=256
    )
    exclude: tuple[str, ...] = Field(default=("**/_partials/**", "**/drafts/**"), max_length=256)
    route_manifest: str | None = Field(default=None, min_length=1, max_length=4_096)
    max_bytes: int | None = Field(default=None, ge=1)

    @field_validator("repository", "revision")
    @classmethod
    def _one_argument(cls, value: str) -> str:
        if value != value.strip() or "\0" in value or "\n" in value:
            raise ValueError("Git repository and revision must each be one command argument")
        return value

    @field_validator("content_root")
    @classmethod
    def _content_root(cls, value: str) -> str:
        return normalize_repository_path(value, allow_root=True)

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: str) -> str:
        return normalize_base_url(value)

    @field_validator("include", "exclude")
    @classmethod
    def _patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(globs.path_glob(value) for value in values))

    @field_validator("route_manifest")
    @classmethod
    def _route_manifest(cls, value: str | None) -> str | None:
        return None if value is None else normalize_repository_path(value)

    def scope_fingerprint(self, source_scope: str) -> str:
        material = json.dumps(
            {
                "source_scope": source_scope,
                "route_policy": "git-site-v1",
                **self.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()


def _normalized_base_url(value: str) -> str:
    """Validate one site root without reflecting its possibly secret input into errors."""
    try:
        parsed = urlsplit(value)
        # ``urlsplit`` delays this validation until the property is read. Do it while validating
        # configuration so a malformed port cannot first fail while building an endpoint later.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("base_url must use a valid host and optional port") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        msg = (
            "base_url must be an absolute http(s) URL with a hostname. Give the site root "
            "including any context path, e.g. 'https://example.atlassian.net/wiki'"
        )
        raise ValueError(msg)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "base_url must not contain credentials; configure authentication separately"
        )
    if parsed.query:
        raise ValueError("base_url must not contain a query string; configure the site root only")
    if parsed.fragment:
        raise ValueError("base_url must not contain a fragment; configure the site root only")
    return f"{parsed.scheme.lower()}://{parsed.netloc}{parsed.path.rstrip('/')}"


class ConfluenceSnapshotConfig(BaseModel):
    """Configuration for the offline Confluence-snapshot connector.

    Set under ``plugins.config."connector.confluence-snapshot"``. One setting, because the
    input is a directory of page snapshots and everything else about a page is in its
    manifest — a connector option that overrode a manifest would be a second authority for a
    fact the manifest already states, and the two would disagree the first time either was
    edited.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(
        default="",
        description="Directory holding page snapshots, at any depth. Resolved to an absolute "
        "path, because a relative one changes with the working directory — and while a "
        "snapshot's identity is its page id rather than its path, the root is what bounds "
        "every read this connector is allowed to make.",
    )


class FilesystemConfig(BaseModel):
    """Configuration for :class:`~manicule.connectors.filesystem.FilesystemConnector`.

    Set under ``plugins.config."connector.filesystem"`` for a configured source. ``manicule
    index <path>`` builds one directly instead, because the path is the argument rather than a
    setting — and it goes through the same class, so the two cannot behave differently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(
        default="",
        min_length=0,
        description="Directory or file to index. Resolved to an absolute path, because a "
        "document's identity includes its source id and a relative one would change with the "
        "working directory.",
    )
    include_hidden: bool = Field(
        default=False,
        description="Whether to walk dot-files. Off, because the usual dot-file in a "
        "repository is tool state rather than a document.",
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Refuse a file larger than this at discovery, before it is read.",
    )
    exclude: tuple[str, ...] = Field(
        default=(),
        max_length=256,
        description="Root-relative POSIX globs this source never walks. A pattern naming a "
        "directory excludes everything beneath it, and ``archive`` and ``archive/**`` name the "
        "same subtree. Applied by the walk, so discovery and reconciliation refuse the same "
        "paths rather than disagreeing about what this source holds.",
    )
    """Content that must stay where it is and must not be searchable.

    **Never indexed, rather than not in this collection.** The other way to keep something out
    of a result set is a collection rule, and it answers a different question: a rule says what
    a group is, so a document it does not select is still in the corpus and still returned by an
    unscoped search. This says the file is not the corpus. The distinction matters for the case
    it was written for — superseded material kept deliberately, which has to remain in the
    directory because it is the record of what was tried, and must never rank beside what
    replaced it.

    **It stops a document being indexed; deleting one already indexed is reconciliation's job,
    not this setting's.** The walk is where this is enforced, so a file that starts matching
    simply stops being discovered. A document that was indexed before the pattern was written
    stays in the index, served and stale, until a reconciliation pass runs: the pass enumerates
    through the same walk, so the file is absent from its inventory and is soft-deleted. A served
    manicule runs that pass on a schedule and after every full sync, and
    ``manicule connector reconcile`` runs it now. Excluding a large share of a source trips the
    deletion ceiling (``ingest.reconcile_max_delete_fraction``), and the proposal then waits for
    ``--confirm``. Said here rather than left to be discovered, because "I excluded it and it is
    still in my results" is the first thing this setting will be blamed for, and the answer is
    a pass rather than a defect.

    **Empty by default, where :class:`GitSiteConfig` ships opinions.** A website has parts that
    are structurally not pages; a directory somebody pointed this connector at is the corpus,
    and a default exclusion would be this module deciding that some of it does not count.
    :data:`~manicule.connectors.filesystem.IGNORED_DIRECTORIES` already covers the one case that
    is never a document anywhere — version control and tool output — and it is not configuration
    because nobody is indexing their ``.git`` on purpose.
    """
    enriched_profiles: tuple[EnrichedProfile, ...] = Field(
        default=(DEFAULT_PROFILE,),
        description="Enriched-export conventions to recognize inside HTML files, in precedence "
        "order. An empty tuple turns adaptation off and indexes every HTML file as HTML.",
    )
    """How a site whose exporter spells the markers differently teaches this connector to read it.

    **Configuration rather than code, and validated rather than trusted.** Every field of a
    profile is checked when the settings are loaded
    (:class:`~manicule.connectors.enriched.EnrichedProfile`): a selector that is not an attribute
    selector is refused, because an element name would make an ordinary ``<main>`` a storage body
    and that is the guess this whole mechanism exists instead of; a representation outside the
    allowlist is refused, because a profile's representation decides which parser untrusted
    extracted markup reaches; and a label mapped to a field that does not exist is refused rather
    than ignored, because an alias that silently does nothing is indistinguishable from not having
    written it.

    Replacing the default rather than adding to it is the caller's choice to make. Listing
    :data:`~manicule.connectors.enriched.DEFAULT_PROFILE` alongside a new one keeps both; omitting
    it means this corpus is not the default form and should not be searched for it.
    """

    @field_validator("exclude")
    @classmethod
    def _patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(globs.path_glob(value) for value in values))


class MboxConfig(BaseModel):
    """Configuration for :class:`~manicule.connectors.mbox.MboxConnector`.

    Every other key is refused, which is how the settings that are deliberately absent (a date
    range, a folder list, a ``Message-ID`` identity) stay absent: a filter evaluated on a full
    walk and not on a re-walk is the failure ``docs/connectors/gmail.md`` §3 states for labels,
    and a path glob already says which files count (``docs/connectors/mbox.md`` §1).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(
        default="",
        description="A directory to search for mbox files, or one mbox file. Resolved to an "
        "absolute path, because a document's address is built from where its file is.",
    )
    include_hidden: bool = Field(
        default=False,
        description="Whether to walk dot-files and dot-directories. Off, as for the filesystem "
        "connector.",
    )
    exclude: tuple[str, ...] = Field(
        default=(),
        max_length=256,
        description="Root-relative POSIX globs this source never walks, read exactly as the "
        "filesystem connector reads them. Applied by the walk, so discovery and reconciliation "
        "refuse the same paths.",
    )
    skip_known_message_ids: bool = Field(
        default=True,
        description="Skip a message whose Message-ID another mail source (a configured gmail or "
        "mbox instance) already holds, unless this source holds it already "
        "(docs/connectors/mbox.md §6). A message with no Message-ID is never skipped.",
    )

    @field_validator("exclude")
    @classmethod
    def _patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(globs.path_glob(value) for value in values))

    def scope_fingerprint(self, source_scope: str) -> str:
        """The scope a stored per-file watermark is valid for.

        Everything that changes which messages discovery yields is in it, so editing any option
        discards the stored file map and the next sync reads every file again — which is what
        brings back the messages ``skip_known_message_ids`` had skipped.
        """
        material = json.dumps(
            {
                "source_scope": source_scope,
                "include_hidden": self.include_hidden,
                "exclude": sorted(self.exclude),
                "skip_known_message_ids": self.skip_known_message_ids,
                "representation": MBOX_REPRESENTATION_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()


class Deployment(StrEnum):
    """Which Confluence this is, which decides both the auth header and the body format.

    Declared rather than probed. A connector that sniffed the deployment would decide it once,
    from whichever endpoint happened to answer first, and a wrong guess is not visible in the
    result — it is visible three stages later as a corpus of pages whose structure was thrown
    away.
    """

    CLOUD = "cloud"
    """Atlassian-hosted. Bodies in Atlassian Document Format."""

    SERVER = "server"
    """Server or Data Center. Bodies in storage format."""


class FullInventoryAuthority(StrEnum):
    """The source resource trusted for whole-space full membership."""

    SEARCH = "search"
    """CQL content search, retained as the compatibility default."""

    DIRECT_CURRENT_CONTENT = "direct_current_content"
    """Server/Data Center's direct ``status=current`` content inventory."""


class AuthMethod(StrEnum):
    """What a request proves this account with — a separate question from which Confluence it is.

    These were one question until browser sessions arrived, because each deployment had exactly
    one credential. They are not one question: a self-hosted instance behind an identity
    provider is Server or Data Center for every purpose that decides how a *body* is read, and
    frequently has personal access tokens disabled by policy, so the credential it can offer is
    the browser session its users already hold.

    Splitting the axis is what keeps :class:`Deployment` meaning one thing. A third deployment
    value would have made ``server`` and ``server-behind-sso`` two spellings of the same body
    format, and every branch that reads a body would then have had to know about both.
    """

    API_TOKEN = "api_token"  # noqa: S105 - the name of a credential kind, not a credential
    """Cloud. ``email:token`` over Basic; the token alone authenticates as nobody."""

    PERSONAL_ACCESS_TOKEN = "personal_access_token"  # noqa: S105 - a kind, not a credential
    """Server or Data Center. A scoped token, sent as a Bearer credential."""

    BROWSER_SESSION = "browser_session"
    """Server or Data Center. The cookies of a session a person signed in to in their browser.

    Unlike the other two this is not a static string manicule can hold indefinitely: it expires
    on the instance's schedule, it is renewed out of band, and it carries the whole of that
    person's identity rather than a narrow grant. Everything about how it is captured, stored,
    consulted and retired follows from those three facts —
    :mod:`manicule.connectors.credentials` and :mod:`manicule.connectors.sessions`.
    """


class ConfluenceConfig(BaseModel):
    """Configuration for :class:`~manicule.connectors.confluence.ConfluenceConnector`.

    Set under ``plugins.config."connector.confluence"``.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    base_url: str = Field(
        min_length=1,
        description="Site root including any context path: "
        "``https://example.atlassian.net/wiki`` for Cloud, ``https://wiki.example.com`` or "
        "``https://example.com/confluence`` for Server. Every request and every link the "
        "source returns is checked against this origin.",
    )

    deployment: Deployment = Deployment.CLOUD

    auth: AuthMethod | None = Field(
        default=None,
        description="How to authenticate. Unset means the deployment's usual credential — an "
        "API token for Cloud, a personal access token for Server and Data Center — so an "
        "existing configuration keeps working without naming it. Set it to "
        "``browser_session`` for an instance whose identity provider makes personal access "
        "tokens unobtainable.",
    )
    """Read through :attr:`auth_method`, which resolves the ``None`` default. It is kept
    nullable rather than defaulted per deployment because the default is a function of another
    field, and a stored ``api_token`` that silently meant something else after ``deployment``
    changed would be a configuration that lies about itself."""

    session_max_age_hours: float = Field(
        default=12.0,
        gt=0.0,
        description="How long after capture manicule will keep using a browser session before "
        "refusing it and asking for a fresh sign-in.",
    )
    """A session cookie carries no expiry manicule can read: the instance decides when it dies
    and says so only by answering a request with a sign-in page. This is manicule's own ceiling
    rather than the instance's, and it exists so that a dead session is a **startup** refusal
    naming what to do, rather than something discovered at the first page of a sync. Set it near
    the identity provider's own session lifetime; too high costs nothing but a later, noisier
    failure, and too low costs an unnecessary sign-in."""

    browser_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="How long `manicule connector login --browser` waits for a person to finish "
        "signing in before giving up and storing nothing.",
    )
    """Five minutes, because the thing being waited for is a person and an identity provider.

    A flat field rather than a ``[connectors.<name>.browser_auth]`` table: this is the only
    browser setting there is, and a nested table for one value invites the next one to go there
    rather than being thought about. ``--timeout`` overrides it per run, which is what somebody
    reaches for when a conditional-access check turns out to need longer than the default."""

    # There is no `session_env`, and its absence is deliberate. It named an environment
    # variable a browser session could be read from, for the platforms that had no macOS
    # Keychain — which is to say it offered a live corporate credential written into a shell's
    # history, visible in every process listing that inherited it, as the *recommended* answer
    # for Linux and containers. Sessions are now held in the running server's memory, which is
    # an answer that is the same on every platform, so the variable is deleted rather than kept
    # as a fallback. `extra="forbid"` means a configuration still setting it is refused loudly
    # instead of silently ignored, which is the choice #98 made for `schedule_s`.
    #
    # The credential is never read from configuration either. `extra="forbid"` means a
    # `session_cookie` key in `config.toml` is a startup error rather than a working setting,
    # and that is deliberate: a session cookie is the sync account's whole identity, and a
    # configuration file is a file that ends up in version control eventually.

    email: str = Field(
        default="",
        description="Cloud only: the account the API token belongs to. Basic auth is "
        "``email:token``, and a token without its email authenticates as nobody.",
    )

    api_token: SecretStr | None = Field(
        default=None,
        description="Cloud API token. Resolved from the environment when unset — see "
        "``token_env``. Held as a secret so printing configuration cannot leak it.",
    )

    personal_access_token: SecretStr | None = Field(
        default=None,
        description="Server/Data Center personal access token, sent as a Bearer credential.",
    )

    token_env: str = Field(
        default="CONFLUENCE_API_TOKEN",
        min_length=1,
        description="Environment variable consulted when no token is set here. A credential "
        "in a configuration file is a credential in version control eventually.",
    )

    spaces: tuple[str, ...] = Field(
        default=(),
        description="Space keys to sync. Empty means every space the account can see, "
        "enumerated at the start of each run so a space created since the last sync is "
        "picked up without a configuration change.",
    )

    root_page_ids: tuple[str, ...] = Field(
        default=(),
        description="Page ids whose trees are the scope of this source. Empty means whole "
        "spaces, which is the behavior every existing configuration has. Set it to index one "
        "documentation area rather than everything its space contains.",
    )

    """A **narrowing of** :attr:`spaces`, never a second way of widening it.

    The two settings answer one question between them: ``spaces`` says which spaces this source
    may read at all, and ``root_page_ids`` says which trees inside them it actually reads. So
    the effective scope is the intersection, and every combination that cannot be honored as an
    intersection is refused at startup rather than resolved by a rule nobody would guess — a
    root outside the allowlist, and a listed space containing no configured root, are both
    configuration errors. Union would make ``spaces`` stop being an allowlist the moment a root
    was added outside it, which is the one thing a setting called an allowlist may not do.

    With no ``spaces`` there is no allowlist, and each root's space is read from the root
    itself. That is the shortest correct configuration for "index this page tree": one setting.
    """

    include_root_pages: bool = Field(
        default=True,
        description="Whether the configured root pages are themselves indexed, alongside their "
        "descendants.",
    )
    """**True**, because ``root_page_ids = ["100100"]`` names page 100100 and a corpus that
    contains everything under it *except it* is not what anybody wrote that down to mean.

    The default decides what a corpus contains, so it is chosen for the direction the mistake
    fails in. Defaulting to ``false`` would leave exactly one page missing — the one that was
    named — and nothing about the run would say so: the counts look right, the descendants are
    all there, and the gap surfaces months later as a citation to an overview page that is not
    in the index. Defaulting to ``true`` costs one extra page for somebody who wanted only the
    children, and they find out immediately, because the page they did not want is the first
    thing they see.

    Setting it with no ``root_page_ids`` is refused rather than ignored: with no roots there is
    no root page to include or leave out, and a setting that silently does nothing reads as one
    that is in force.
    """

    full_inventory_authority: FullInventoryAuthority = Field(
        default=FullInventoryAuthority.SEARCH,
        description="Membership authority for complete Server/Data Center whole-space walks. "
        "The direct strategy uses /rest/api/content with status=current; incremental and "
        "subtree discovery remain CQL-backed.",
    )

    page_size: int = Field(
        default=100,
        ge=1,
        le=250,
        description="Results per search request. The ceiling is the source's own; asking for "
        "more is silently reduced, which makes a sync's request count differ from what its "
        "configuration says.",
    )

    adaptive_min_page_size: int = Field(
        default=1,
        ge=1,
        le=250,
        description="Smallest page the authoritative direct inventory will shrink to when a "
        "request times out at an offset. The floor of the halving policy, never the start.",
    )
    """Applies only to the Server/Data Center ``direct_current_content`` walk, where response
    time grows with the offset and a full page can exceed the request timeout while a smaller
    page at the same offset still answers. Shrinking changes nothing about membership: the
    same rows arrive in the same order across more, smaller requests. Deliberately absent from
    the scope fingerprint for exactly that reason — retuning it must not force a full
    re-enumeration."""

    adaptive_max_attempts_per_offset: int = Field(
        default=8,
        ge=1,
        description="How many timed-out requests one offset may absorb before the run stops "
        "with a typed incomplete result.",
    )
    """Eight covers the default halving ladder (100 → 50 → 25 → 12 → 6 → 3 → 1) with slack.
    When it is spent the original timeout is raised, the durable prefix is kept, and nothing
    is promoted — a slow offset is never allowed to read as the end of the inventory."""

    adaptive_max_seconds_per_offset: float = Field(
        default=240.0,
        gt=0.0,
        description="Longest cumulative time the adaptive policy may spend at one offset "
        "before giving up with a typed incomplete result.",
    )
    """The wall-clock twin of ``adaptive_max_attempts_per_offset``: attempts bound the count,
    this bounds the stall an operator watches. Both are ceilings on one offset rather than on
    the run, because the run's length is the corpus's business."""

    adaptive_page_size_growth: bool = Field(
        default=False,
        description="Whether a successful page after a timeout shrink may cautiously grow the "
        "requested size again, doubling per successful page back toward page_size.",
    )
    """Off by default. The observed failure is *progressive* large-offset latency, where a
    regrown page re-times-out a few offsets later and every recovery costs a full request
    timeout of stall. Turn it on for a source whose slowness is patchy rather than
    progressive."""

    watermark_overlap_minutes: int = Field(
        default=5,
        ge=0,
        description="How far back before the stored watermark each incremental query starts.",
    )
    """CQL's ``lastmodified`` compares at minute granularity, so a page saved at 14:30:59 and
    one saved at 14:30:01 are indistinguishable to it. Re-enumerating a few minutes of overlap
    costs one comparison per page — change detection skips anything whose version token is
    unchanged before it is fetched (``docs/ingest.md`` §4) — while missing a page costs a
    document that is wrong in the index until something unrelated happens to touch it."""

    resolve_macros: bool = Field(
        default=True,
        description="Expand ``include`` and ``excerpt-include`` macros into the page body.",
    )
    """Unresolved, that content is missing from the chunk while appearing present in the UI —
    the failure is invisible from both ends (``docs/connectors/confluence.md`` §5). Turning
    this off does not make it quiet: every macro left unexpanded is recorded on the document."""

    macro_depth: int = Field(
        default=3,
        ge=0,
        le=10,
        description="How far macro expansion may nest. A page including a page that includes "
        "a page is real; deeper is a template loop.",
    )

    include_attachments: bool = Field(
        default=True,
        description="Discover attachments as documents of their own, routed through the "
        "normal parser chain — a PDF attached to a page is a PDF.",
    )

    max_attachment_bytes: int = Field(
        default=64 * 1024**2,
        gt=0,
        description="Largest attachment downloaded. Enforced while streaming rather than "
        "against the declared size, because the declared size is the source's claim.",
    )

    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    max_retries: int = Field(
        default=5,
        ge=0,
        description="Attempts after the first for a throttled, failed or interrupted request.",
    )

    max_retry_after_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Longest single pause honored from a ``Retry-After`` header.",
    )
    """A sync that sleeps for an hour inside one HTTP call is indistinguishable from a hung
    one. Past this the connector gives up and says so, and the run is re-run later against an
    unadvanced watermark, which costs re-enumeration and nothing else."""

    cursor_lifetime_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="How long a pagination cursor may be held before the connector refuses "
        "to use it.",
    )
    """Search cursors expire (``docs/connectors/confluence.md`` §2). A consumer that stalls
    mid-enumeration on source traffic, journal admission or a paused process resumes onto a
    cursor the server has forgotten, and a forgotten cursor can be answered with a fresh first
    page rather than an error. That enumerates the opening of the corpus twice and its tail
    never, silently. With acquisition journaling enabled, local parsing and embedding start from
    the durable journal after enumeration and cannot age this cursor. The compatibility fallback
    remains backpressure-coupled. The guard turns any remaining stall into a refusal before the
    request is sent."""

    @model_validator(mode="after")
    def _checkable_settings(self) -> Self:
        self.base_url = _normalized_base_url(self.base_url)
        for space in self.spaces:
            if not space.strip():
                msg = "spaces contains an empty key; remove it rather than syncing everything"
                raise ValueError(msg)
        for root in self.root_page_ids:
            if not is_page_id(root.strip()):
                msg = (
                    f"root_page_ids contains {root!r}, which is not a Confluence page id. Ids "
                    f"are decimal numbers: take the `pageId` from the page's URL, or open the "
                    f"page and use the number at the end of its short link. A title will not "
                    f"do — two pages in one space can share one, and the id is what survives a "
                    f"rename."
                )
                raise ValueError(msg)
        if not self.root_page_ids and "include_root_pages" in self.model_fields_set:
            msg = (
                "include_root_pages is set but root_page_ids is empty, so there is no root "
                "page for it to include or leave out. Set root_page_ids to the tree(s) this "
                "source should index, or remove include_root_pages — a setting that silently "
                "does nothing reads like one that is in force."
            )
            raise ValueError(msg)
        if self.adaptive_min_page_size > self.page_size:
            msg = (
                f"adaptive_min_page_size ({self.adaptive_min_page_size}) is larger than "
                f"page_size ({self.page_size}). The adaptive floor is where a timed-out "
                f"request shrinks *down* to; a floor above the starting size would make the "
                f"first shrink an enlargement."
            )
            raise ValueError(msg)
        if self.auth is AuthMethod.BROWSER_SESSION and self.deployment is not Deployment.SERVER:
            msg = (
                "auth = 'browser_session' is a Server/Data Center arrangement and this is "
                "configured as Cloud. Cloud sessions are held by Atlassian's own domains and "
                "are not a credential a REST client can carry; Cloud authenticates with an API "
                "token created at https://id.atlassian.com/manage-profile/security/api-tokens. "
                "Refused rather than attempted, because an attempt would authenticate as "
                "nobody and index whatever an anonymous reader can see."
            )
            raise ValueError(msg)
        if self.auth is AuthMethod.API_TOKEN and self.deployment is not Deployment.CLOUD:
            msg = (
                "auth = 'api_token' is Cloud's credential and this is configured as "
                "Server/Data Center, which has no API tokens. Use 'personal_access_token', or "
                "'browser_session' if this instance's identity provider has disabled them."
            )
            raise ValueError(msg)
        if (
            self.auth is AuthMethod.PERSONAL_ACCESS_TOKEN
            and self.deployment is not Deployment.SERVER
        ):
            msg = (
                "auth = 'personal_access_token' is a Server/Data Center credential and this is "
                "configured as Cloud, which issues API tokens instead. Use 'api_token' with the "
                "email the token belongs to."
            )
            raise ValueError(msg)
        # Written back deduplicated and in a stable order, because this tuple is half of the
        # scope identity a stored watermark is compared against: `["1", "2"]` and `["2", "1",
        # "1"]` are the same scope, and a run that re-enumerated everything because somebody
        # reordered a list would be a full sync nobody asked for.
        self.root_page_ids = tuple(dict.fromkeys(root.strip() for root in self.root_page_ids))
        return self

    @property
    def current_only(self) -> bool:
        """Whether this deployment's search accepts ``status = current``.

        **The one place the deployment decides this**, read by every query this connector
        builds — whole-space discovery, incremental discovery, page-tree discovery, attachment
        discovery, reconciliation, subtree membership, attachment reconciliation and the title
        lookup an include macro resolves through. Eight call sites, one answer, so there is no
        list of places to remember to keep in step.

        Cloud accepts the field and needs it: reconciliation depends on a trashed page not being
        returned. The standard Data Center content-search resource rejects it and returns
        current content by default, so the same clause is an HTTP 400 there.

        **Read from the declared deployment and from nothing else.** Not the URL shape, not the
        hostname, not the context path, not the authentication method, and above all not from
        catching a 400 and retrying without the clause — a query that is sent to find out
        whether it is valid has already told a wiki something wrong about what manicule wants,
        and the retry would mask the day Atlassian changes which deployments accept it.
        """
        return self.deployment is Deployment.CLOUD

    @property
    def scope_identity(self) -> str:
        """What this configuration's scope is, in a form a stored watermark can be compared to.

        A watermark is a position **within a scope**, and the two are meaningless apart: a
        position recorded while one page tree was configured says nothing about a run that has
        just been pointed at a different one. Reusing it would skip every page in the new scope
        that had not changed since — permanently, because nothing enumerates them again.

        Deliberately legible rather than a digest. It is read by whoever is working out why a
        sync re-enumerated, and ``roots=100100,100200 include_roots=true`` answers that where a
        hash would only prove that something changed.

        Spaces are absent on purpose: they are already the keys of the per-space map, so a space
        added to the allowlist arrives with no stored position and is enumerated in full on its
        own, and one removed simply stops being asked about. Neither needs the rest discarded.
        """
        if not self.root_page_ids:
            return "whole-space"
        roots = ",".join(sorted(self.root_page_ids))
        return f"roots={roots} include_roots={str(self.include_root_pages).lower()}"

    @property
    def effective_full_inventory_authority(self) -> FullInventoryAuthority:
        """Authority that actually changes behavior for this deployment and scope.

        Cloud and subtree scopes retain CQL semantics even when configuration contains the
        direct option.  Returning the effective value keeps those ignored settings out of
        cursor identity and avoids a surprise complete re-enumeration.
        """
        if self.deployment is Deployment.SERVER and not self.root_page_ids:
            return self.full_inventory_authority
        return FullInventoryAuthority.SEARCH

    def scope_fingerprint(self, source_scope: str) -> str:
        """Cursor identity, preserving the historical digest for search-backed scopes."""
        material = source_scope
        if self.effective_full_inventory_authority is FullInventoryAuthority.DIRECT_CURRENT_CONTENT:
            material = f"{material};full_inventory=direct_current_content"
        return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()

    @property
    def auth_method(self) -> AuthMethod:
        """How this configuration authenticates, with the deployment's default filled in.

        Every reader goes through this rather than through :attr:`auth`, so that "unset" is
        resolved in exactly one place instead of once per call site with one of them wrong.
        """
        if self.auth is not None:
            return self.auth
        if self.deployment is Deployment.CLOUD:
            return AuthMethod.API_TOKEN
        return AuthMethod.PERSONAL_ACCESS_TOKEN

    @property
    def origin(self) -> str:
        """Scheme, host and port of :attr:`base_url`, for checking links against.

        **Delegated to :func:`~manicule.connectors.pagination.origin_of` rather than derived
        here**, because every caller compares this against that function's output — a link base
        a response declared, a ``webui`` path joined onto one. Two spellings of "the origin"
        agree on the common case and part company on the uncommon one, and the uncommon one here
        was a ``base_url`` with a capital letter in the host or credentials embedded in it: this
        property returned them verbatim, ``origin_of`` lowercased the host and dropped the
        userinfo, and the comparison then declared a same-origin link untrusted. A connector
        that refuses its own instance is a hard failure with a message pointing at the wrong
        cause, so the two readings are now one reading.
        """
        return origin_of(self.base_url)


def resolve_credentials(
    config: ConfluenceConfig, environ: Mapping[str, str] | None = None
) -> ConfluenceConfig:
    """Fill in the credential from the environment, and refuse a configuration that has none.

    Runs in the factory, before the connector is constructed, because a missing credential is
    a misconfiguration and ``CONTRIBUTING.md`` puts those at startup: discovering it at the
    first page of the first sync means a run that reports progress and indexes nothing.

    An explicitly configured token is never overwritten by an environment variable.

    **A browser session is not resolved here and never lands on this model.** It is a secret
    with a lifetime, it lives in the running server's memory rather than in configuration, and
    it is read again for every request rather than once — none of which belongs on a model that
    plugin discovery imports in every process that starts.
    :func:`manicule.connectors.credentials.credential_for` does that half, and the factory calls
    both before constructing anything.

    Raises:
        ConfigError: The credential for this deployment is absent, or Cloud was configured
            without the email its Basic credential needs.
    """
    env = os.environ if environ is None else environ
    resolved = config

    if config.auth_method is AuthMethod.BROWSER_SESSION:
        return resolved

    if config.deployment is Deployment.CLOUD:
        token = config.api_token or _secret(env.get(config.token_env))
        if token is None:
            msg = (
                f"Confluence Cloud needs an API token. Set it in "
                f'plugins.config."connector.confluence".api_token, or put it in '
                f"${config.token_env}. Create one at "
                f"https://id.atlassian.com/manage-profile/security/api-tokens"
            )
            raise ConfigError(msg)
        if not config.email.strip():
            msg = (
                "Confluence Cloud authenticates as 'email:token' over Basic. The token alone "
                "authenticates as nobody, so set "
                'plugins.config."connector.confluence".email to the account the token '
                "belongs to."
            )
            raise ConfigError(msg)
        resolved = config.model_copy(update={"api_token": token})
    else:
        token = config.personal_access_token or _secret(env.get(config.token_env))
        if token is None:
            msg = (
                f"Confluence Server/Data Center needs a personal access token. Set it in "
                f'plugins.config."connector.confluence".personal_access_token, or put it '
                f"in ${config.token_env}. Create one from your profile under Personal Access "
                f"Tokens."
            )
            raise ConfigError(msg)
        resolved = config.model_copy(update={"personal_access_token": token})

    return resolved


_SYSTEM_LABELS = frozenset(label.casefold() for label in ALWAYS_EXCLUDED_LABELS)


class GmailConfig(BaseModel):
    """Configuration for the Gmail connector (``docs/connectors/gmail.md`` §1, §3).

    Set under ``[connectors.<name>.options]`` for a ``type = "gmail"`` source. Printing it never
    shows a secret: both are :class:`~pydantic.SecretStr`, and ``hide_input_in_errors`` keeps a
    rejected value, the mailbox address included, out of a validation message.

    **What is deliberately not a setting** (``gmail.md`` §3, §5), so that its absence reads as a
    decision rather than an omission, and ``extra="forbid"`` turns an attempt to set one into a
    startup error instead of a silently ignored line:

    - No free-form ``query``. Incremental discovery cannot evaluate one, so a query would be
      honored by the full walk and by nothing after it. Label membership is the only scope all
      three discovery paths can test.
    - No ``include_spam_trash``. SPAM, TRASH and DRAFT are always excluded.
    - No ``user``. A grant reads only its own mailbox; ``account`` pins which one that must be.
    - No message or attachment size cap. Size is the pipeline's ``ingest.max_fetch_bytes``.
    - No token endpoint. The refresh goes to Google's, and a setting for it would let one line of
      configuration send the client secret and the refresh token to any host.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    account: str = Field(
        min_length=1,
        description="The one mailbox this instance indexes, as its primary address. A token for "
        "any other mailbox is refused, and editing this on an instance that has synced is "
        "refused too: indexing a different mailbox is a different instance.",
    )
    """**Required, and the primary address.** The token decides which mailbox is read, and a
    token is replaced routinely, so the mailbox is configuration rather than a side effect of
    whichever credential happens to be supplied (``gmail.md`` §1.4). It is compared with what
    ``users.getProfile`` reports, which is the primary address: an alias is refused as a
    mismatch rather than followed. Stored lowercase, because Google compares addresses without
    regard to case and two spellings of one address must not be two scopes."""

    client_id: str = Field(
        min_length=1,
        description="The OAuth client's id. Not a secret.",
    )

    client_secret: SecretStr | None = Field(
        default=None,
        description="The OAuth client's secret. Resolved from the environment when unset — see "
        "``client_secret_env``. Held as a secret so printing configuration cannot leak it.",
    )

    client_secret_env: str = Field(
        default="GMAIL_CLIENT_SECRET",
        min_length=1,
        description="Environment variable consulted when no client secret is set here.",
    )

    refresh_token: SecretStr | None = Field(
        default=None,
        description="The long-lived refresh token ``manicule connector login`` mints. Resolved "
        "from the environment when unset — see ``refresh_token_env``.",
    )

    refresh_token_env: str = Field(
        default="GMAIL_REFRESH_TOKEN",
        min_length=1,
        description="Environment variable consulted when no refresh token is set here. A "
        "credential in a configuration file is a credential in version control eventually, so "
        "this is the documented path.",
    )

    exclude_labels: tuple[str, ...] = Field(
        default=(),
        max_length=256,
        description="Label **names** whose messages are never indexed, resolved to ids through "
        "``users.labels.list``. SPAM, TRASH and DRAFT are always excluded and are not listed "
        "here. A name the mailbox does not have is a configuration error, never an empty "
        "exclusion.",
    )

    quota_units_per_minute: int = Field(
        default=5_400,
        ge=GMAIL_RAW_MESSAGE_UNITS,
        le=GMAIL_QUOTA_UNITS_PER_MINUTE,
        description="The client's own pacing budget: 90% of Gmail's per-user limit of 6,000 "
        "units a minute. Requests wait before they would exceed it, so the client is steered by "
        "its own accounting rather than by 429s.",
    )
    """Bounded on both sides, and for different reasons. Above 6,000 the budget is a promise
    the source will not keep, and the client is back to steering by 429s. Below 60 — the cost of
    one raw message — no fetch could ever be admitted, and the run would wait forever instead of
    saying why."""

    max_retry_after_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Longest single pause honored from a ``Retry-After`` header; a longer "
        "request stops the run.",
    )
    """As :attr:`ConfluenceConfig.max_retry_after_seconds`: a sync that sleeps for an hour inside
    one HTTP call is indistinguishable from a hung one, and a stopped run re-runs against an
    unadvanced watermark."""

    max_retries: int = Field(
        default=5,
        ge=0,
        description="Attempts after the first for a throttled, failed or interrupted request.",
    )

    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("account")
    @classmethod
    def _account(cls, value: str) -> str:
        # The value is never echoed: it is the address this whole connector is careful not to
        # print, and a configuration error is read by a wider audience than the mailbox is.
        address = value.strip().casefold()
        local, separator, domain = address.partition("@")
        if (
            not separator
            or not local
            or not domain
            or "@" in domain
            or any(c.isspace() for c in address)
        ):
            msg = (
                "account must be a mailbox address of the form name@example.org: the primary "
                "address of the Google Workspace account whose mail this instance indexes"
            )
            raise ValueError(msg)
        return address

    @field_validator("client_id")
    @classmethod
    def _client_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("client_id is blank; set the OAuth client's id")
        return value.strip()

    @field_validator("exclude_labels")
    @classmethod
    def _exclude_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        names: list[str] = []
        for raw in values:
            name = raw.strip()
            if not name:
                msg = (
                    "exclude_labels contains an empty name; remove it rather than excluding nothing"
                )
                raise ValueError(msg)
            if name.casefold() in _SYSTEM_LABELS:
                msg = (
                    f"exclude_labels lists {name!r}, which is always excluded. SPAM, TRASH and "
                    f"DRAFT cannot be brought back into scope, so listing one does nothing; "
                    f"remove it — a setting that silently does nothing reads like one in force."
                )
                raise ValueError(msg)
            names.append(name)
        return tuple(dict.fromkeys(names))

    @model_validator(mode="after")
    def _secrets_not_blank(self) -> Self:
        for field_name in ("client_secret", "refresh_token"):
            secret = getattr(self, field_name)
            if secret is not None and not secret.get_secret_value():
                msg = (
                    f"{field_name} is set to an empty string. Leave it unset to read "
                    f"${getattr(self, field_name + '_env')}, or give it a value."
                )
                raise ValueError(msg)
        return self

    @property
    def pinned_identity(self) -> str:
        """A digest of the configured mailbox, for the pipeline to compare across runs.

        A digest rather than the address, because it is stored beside the instance's run
        metadata, which status surfaces read, and those carry no address (``gmail.md`` §10).
        Independent of :meth:`scope_fingerprint` on purpose: a scope change keeps the source and
        re-walks it, and a different mailbox must never be treated as one.
        """
        return hashlib.blake2b(f"gmail-account:{self.account}".encode(), digest_size=20).hexdigest()

    def scope_fingerprint(self, source_scope: str, excluded_label_ids: Iterable[str]) -> str:
        """The scope a stored history position is valid for.

        Takes the **resolved** label ids rather than the configured names, because ids are what
        every discovery path tests (``gmail.md`` §3) and a label renamed in Gmail is the same
        scope. Resolving them needs ``users.labels.list``, which is the REST client's call, so
        the connector resolves first (:func:`resolve_label_ids`) and passes the result.

        ``account`` and the representation version are in it as well, so state recorded for one
        mailbox is never resumed for another even if the credential check were bypassed, and a
        change to what is fetched discards the history position (``gmail.md`` §1.4, §4.1).
        """
        material = json.dumps(
            {
                "source_scope": source_scope,
                "account": self.account,
                "excluded_label_ids": sorted(set(excluded_label_ids)),
                "representation": REPRESENTATION_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()


def resolve_label_ids(names: Iterable[str], name_to_id: Mapping[str, str]) -> tuple[str, ...]:
    """The label ids for configured label names, sorted and without duplicates.

    Pure, so the rule is testable without a mailbox: ``name_to_id`` is whatever
    ``users.labels.list`` returned. Matching is exact, because a near match is a guess about
    which mail to keep out.

    Raises:
        ConfigError: A name is not in ``name_to_id``. It names the **configured** name, which is
            the operator's own text, and never lists the labels the mailbox has: label names are
            mail-derived, and run-level errors carry none (``gmail.md`` §3, §10). Refused rather
            than skipped, because a typo that silently excluded nothing would index precisely the
            mail the operator meant to keep out.
    """
    ids: set[str] = set()
    for name in names:
        label_id = name_to_id.get(name)
        if label_id is None:
            msg = (
                f"exclude_labels names {name!r}, which is not a label in this mailbox. Names "
                f"are matched exactly, nested labels are written with their slashes "
                f"('Parent/Child'), and a label that does not exist excludes nothing — which "
                f"is why it is refused rather than ignored."
            )
            raise ConfigError(msg)
        ids.add(label_id)
    return tuple(sorted(ids))


def resolve_gmail_client_secret(
    config: GmailConfig, environ: Mapping[str, str] | None = None, *, instance: str = ""
) -> SecretStr:
    """The OAuth client's secret: the configured value, else the environment variable.

    Split out of :func:`resolve_gmail_credentials` because ``manicule connector login`` needs
    the client secret and, by definition, has no refresh token yet — resolving both together
    would refuse the one command that produces the second.

    Raises:
        ConfigError: Neither is set. The message names the setting and the environment
            variable, and never a value.
    """
    env = os.environ if environ is None else environ
    client_secret = config.client_secret or _secret(env.get(config.client_secret_env))
    if client_secret is None:
        where = _gmail_where(instance)
        msg = (
            f"the Gmail connector needs the OAuth client's secret. Set client_secret in {where}, "
            f"or put it in ${config.client_secret_env}. It is the secret of the Desktop-app "
            f"OAuth client created in docs/connectors/gmail.md §1.3."
        )
        raise ConfigError(msg)
    return client_secret


def resolve_gmail_credentials(
    config: GmailConfig, environ: Mapping[str, str] | None = None, *, instance: str = ""
) -> GmailConfig:
    """Fill in the client secret and refresh token from the environment, refusing any gap.

    Runs in the factory, before the connector is constructed, on the principle
    :func:`resolve_credentials` states: a credential problem found at the first request of the
    first sync is a run that reports progress and indexes nothing. An explicitly configured
    value is never overwritten by an environment variable, and a blank variable counts as unset.

    Args:
        config: The validated settings. ``account`` and ``client_id`` are required by the model,
            so the container has already named them if either was missing.
        environ: The environment to read; the process's own by default.
        instance: The configured source's name, so the message points at the table the setting
            belongs in rather than at the global slot.

    Raises:
        ConfigError: The client secret or the refresh token is absent. The message names the
            setting and the environment variable, and never a value.
    """
    env = os.environ if environ is None else environ
    client_secret = resolve_gmail_client_secret(config, env, instance=instance)
    refresh_token = config.refresh_token or _secret(env.get(config.refresh_token_env))
    if refresh_token is None:
        msg = (
            f"the Gmail connector needs a refresh token. Set refresh_token in "
            f"{_gmail_where(instance)}, or put it in ${config.refresh_token_env}. "
            f"`manicule connector login {instance or '<name>'}` mints one."
        )
        raise ConfigError(msg)
    return config.model_copy(
        update={"client_secret": client_secret, "refresh_token": refresh_token}
    )


def _gmail_where(instance: str) -> str:
    """The table a Gmail setting belongs in, for the message that asks for it."""
    return f"[connectors.{instance}.options]" if instance else 'plugins.config."connector.gmail"'


def _secret(value: str | None) -> SecretStr | None:
    """A secret from an environment variable, treating blank as absent.

    An empty variable is how a shell reports one that was never set in the file it sourced,
    and authenticating with an empty password produces a 401 that reads like a wrong password.
    """
    return SecretStr(value) if value else None
