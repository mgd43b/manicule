"""What registration needs to know about the Confluence connector, without importing it.

The same split the parsers make (:mod:`manicule.parsers.config`): registration needs the
configuration model eagerly, so settings written for the connector are validated rather than
silently ignored, and nothing else. This module imports nothing heavier than pydantic, so
plugin discovery — which runs in every process that starts, before any configuration is
read — does not load an HTTP client on a machine that is never going to sync anything.

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

from manicule.core.errors import ConfigError

__all__ = [
    "CONNECTOR_NAME",
    "ConfluenceConfig",
    "Deployment",
    "resolve_credentials",
]

CONNECTOR_NAME = "confluence"
"""The registered name of the connector, and what configuration selects it by."""


class Deployment(StrEnum):
    """Which Confluence this is, which decides both the auth header and the body format.

    Declared rather than probed. A connector that sniffed the deployment would decide it once,
    from whichever endpoint happened to answer first, and a wrong guess is not visible in the
    result — it is visible three stages later as a corpus of pages whose structure was thrown
    away.
    """

    CLOUD = "cloud"
    """Atlassian-hosted. Email plus API token, Basic, and bodies in Atlassian Document Format."""

    SERVER = "server"
    """Server or Data Center. Personal access token, Bearer, and bodies in storage format."""


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
        description="Longest single pause honoured from a ``Retry-After`` header.",
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
        return self

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

    Raises:
        ConfigError: The credential for this deployment is absent, or Cloud was configured
            without the email its Basic credential needs.
    """
    env = os.environ if environ is None else environ
    resolved = config

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
