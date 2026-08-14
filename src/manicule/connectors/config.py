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

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from manicule.connectors.cql import is_page_id
from manicule.connectors.enriched import DEFAULT_PROFILE, EnrichedProfile
from manicule.core.errors import ConfigError

__all__ = [
    "CONNECTOR_NAME",
    "FILESYSTEM_CONNECTOR_NAME",
    "SNAPSHOT_CONNECTOR_NAME",
    "AuthMethod",
    "ConfluenceConfig",
    "ConfluenceSnapshotConfig",
    "Deployment",
    "EnrichedProfile",
    "FilesystemConfig",
    "resolve_credentials",
]

CONNECTOR_NAME = "confluence"
"""The registered name of the connector, and what configuration selects it by."""

FILESYSTEM_CONNECTOR_NAME = "filesystem"
"""The registered name of the local-directory connector."""

SNAPSHOT_CONNECTOR_NAME = "confluence-snapshot"
"""The registered name of the offline Confluence-snapshot connector.

A name of its own rather than a mode of ``confluence``, because the two share no
configuration: this one has no base URL, no credential and no deployment, and folding them
together would mean a config model where over half the fields are refused depending on
another field's value. It also keeps the credential refusal honest — a connector that
reaches no network cannot be misconfigured into trying.
"""


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

    model_config = ConfigDict(extra="forbid")

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

    page_size: int = Field(
        default=100,
        ge=1,
        le=250,
        description="Results per search request. The ceiling is the source's own; asking for "
        "more is silently reduced, which makes a sync's request count differ from what its "
        "configuration says.",
    )

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
    mid-enumeration — a slow embed, a paused pipeline — resumes onto a cursor the server has
    forgotten, and a forgotten cursor can be answered with a fresh first page rather than an
    error. That enumerates the opening of the corpus twice and its tail never, silently. This
    turns it into a refusal before the request is sent."""

    @model_validator(mode="after")
    def _checkable_settings(self) -> Self:
        if not self.base_url.startswith(("http://", "https://")):
            msg = (
                f"base_url must be an absolute http(s) URL, got {self.base_url!r}. Give the "
                f"site root including any context path, e.g. "
                f"'https://example.atlassian.net/wiki'"
            )
            raise ValueError(msg)
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
        """Scheme, host and port of :attr:`base_url`, for checking links against."""
        without_scheme = self.base_url.split("://", 1)
        scheme = without_scheme[0]
        rest = without_scheme[1] if len(without_scheme) > 1 else ""
        return f"{scheme}://{rest.split('/', 1)[0]}"


def resolve_credentials(
    config: ConfluenceConfig, environ: Mapping[str, str] | None = None
) -> ConfluenceConfig:
    """Fill in the credential from the environment, and refuse a configuration that has none.

    Runs in the factory, before the connector is constructed, because a missing credential is
    a misconfiguration and ``CONTRIBUTING.md`` puts those at startup: discovering it at the
    first page of the first sync means a run that reports progress and indexes nothing.

    An explicitly configured token is never overwritten by an environment variable.

    **A browser session is not resolved here and never lands on this model.** It is a secret
    with a lifetime, it lives in the operating system's keychain rather than in configuration,
    and reading it costs a subprocess — none of which belongs in a module that plugin discovery
    imports in every process that starts. :func:`manicule.connectors.credentials.credential_for`
    does that half, and the factory calls both before constructing anything.

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


def _secret(value: str | None) -> SecretStr | None:
    """A secret from an environment variable, treating blank as absent.

    An empty variable is how a shell reports one that was never set in the file it sourced,
    and authenticating with an empty password produces a 401 that reads like a wrong password.
    """
    return SecretStr(value) if value else None
