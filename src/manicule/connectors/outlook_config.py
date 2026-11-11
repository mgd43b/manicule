"""What registration needs to know about the Outlook connector, without importing it.

The split :mod:`manicule.connectors.config` makes for the others: registration needs the
configuration model eagerly so settings written for the connector are validated rather than
silently ignored, and it imports nothing heavier than pydantic, so plugin discovery does not load
an HTTP client on a machine that never syncs anything.

``docs/connectors/outlook.md`` is the design. **A Microsoft 365 mailbox and calendar, read-only,
through Microsoft Graph, signed in as a public client.** There is no client secret anywhere in
this connector: the Entra app registration is a public client (a "Mobile and desktop
application"), its id is configuration and not a secret, and the one secret is the refresh token.

**What is deliberately not a setting**, so that its absence reads as a decision, and
``extra="forbid"`` turns an attempt to set one into a startup error:

- No authority host and no tenant alias. The tenant is a directory GUID, and ``common``,
  ``organizations`` and ``consumers`` are refused by name: they let whichever account signs in
  choose the tenant, and an instance indexes one mailbox in one directory. The token endpoint is
  a constant built from the GUID, so a setting cannot send the refresh token to another host.
- No Graph base URL. Public cloud only (``https://graph.microsoft.com/v1.0``); the national
  clouds are different hosts and are not supported rather than guessed at.
- No free-form ``$filter`` or ``$search``. Folder membership is the one scope the full walk, the
  incremental walk and reconciliation can all test.
- No ``include_deleted`` or ``include_junk``. Junk Email, Deleted Items, Drafts, Outbox and Sync
  Issues are always excluded, and so are their subfolders.
- No message or attachment size cap. Size is the pipeline's ``ingest.max_fetch_bytes``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from manicule.core.errors import ConfigError

__all__ = [
    "ALWAYS_EXCLUDED_FOLDERS",
    "EVENT_REPRESENTATION_VERSION",
    "MAIL_REPRESENTATION_VERSION",
    "MAX_CONCURRENT_REQUESTS",
    "OUTLOOK_CONNECTOR_NAME",
    "REFUSED_TENANTS",
    "OutlookConfig",
    "read_token_file",
    "resolve_folder_ids",
    "resolve_outlook_credentials",
]

OUTLOOK_CONNECTOR_NAME: Final = "outlook"
"""The registered name of the Outlook connector."""

MAIL_REPRESENTATION_VERSION: Final = "mime/1"
"""What a message is fetched as (``$value``, the whole MIME message), as a version token.

A message's bytes do not change after delivery, so every message carries this one token and a
stored message is never fetched again by id. It is part of the scope fingerprint for the reason
Gmail's is (``docs/connectors/gmail.md`` §4.1): bumping it discards every stored delta link, so the
next sync walks everything and yields each message at the new token.
"""

EVENT_REPRESENTATION_VERSION: Final = "ics/1"
"""How an event is rendered to ``text/calendar``. Part of the scope fingerprint and of every
event's version token (beside Graph's ``changeKey``), so a change to the rendering re-fetches
every event."""

ALWAYS_EXCLUDED_FOLDERS: Final = (
    "junkemail",
    "deleteditems",
    "drafts",
    "outbox",
    "syncissues",
)
"""Graph's well-known folder names that no configuration can bring back into scope.

Subfolders go with them: Sync Issues holds Conflicts, Local Failures and Server Failures, and
a user's own subfolders of Deleted Items are still deleted mail.
"""

MAX_CONCURRENT_REQUESTS: Final = 4
"""Outlook's own cap: four concurrent requests per app and mailbox. A fifth is throttled."""

REFUSED_TENANTS: Final = frozenset({"common", "organizations", "consumers"})
"""Aliases that mean "whichever directory the account signs in from"."""

_GUID: Final = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

_FOLDER_PATH_SEPARATOR: Final = "/"


class OutlookConfig(BaseModel):
    """Configuration for the Outlook connector (``docs/connectors/outlook.md`` §1, §3).

    Set under ``[connectors.<name>.options]`` for a ``type = "outlook"`` source. Printing it never
    shows a secret: the refresh token is a :class:`~pydantic.SecretStr`, and
    ``hide_input_in_errors`` keeps a rejected value, the mailbox address included, out of a
    validation message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    account: str = Field(
        min_length=1,
        description="The one mailbox this instance indexes: its sign-in name (user principal "
        "name) or its primary address. A token for any other mailbox is refused, and editing "
        "this on an instance that has synced is refused too.",
    )
    """**Required.** The token decides which mailbox is read and a token is replaced routinely, so
    the mailbox is configuration (``outlook.md`` §1.4). It is compared, without regard to case,
    with both the ``userPrincipalName`` and the ``mail`` that ``/me`` reports, because in
    Microsoft 365 the two differ for many accounts and a configuration that names either is
    unambiguous. An alias is not accepted."""

    tenant_id: str = Field(
        description="The Microsoft Entra directory (tenant) id, a GUID. The aliases `common`, "
        "`organizations` and `consumers` are refused.",
    )

    client_id: str = Field(
        description="The Entra app registration's application (client) id, a GUID. Not a secret. "
        "The registration must be a public client with a `http://localhost` redirect URI.",
    )

    refresh_token: SecretStr | None = Field(
        default=None,
        description="The refresh token `manicule connector login` mints. Resolved from "
        "`refresh_token_file`, then the environment, when unset.",
    )

    refresh_token_env: str = Field(
        default="OUTLOOK_REFRESH_TOKEN",
        min_length=1,
        description="Environment variable consulted when no token or file supplies one.",
    )

    refresh_token_file: str | None = Field(
        default=None,
        min_length=1,
        description="A file holding the refresh token. **Microsoft rotates refresh tokens**: each "
        "refresh returns a replacement, and the old one is meant to be discarded. When this is "
        "set, the replacement is written back here (a new file readable only by you, renamed "
        "into place), so the token keeps its lifetime. Without it a rotated token lives in "
        "memory and is lost when the process stops.",
    )

    login_redirect_host: Literal["localhost", "127.0.0.1"] = Field(
        default="localhost",
        description="The host name `connector login` puts in its redirect address. Entra ignores "
        "the port of a loopback redirect, and the portal only registers `http://localhost`; use "
        "`127.0.0.1` only if the app manifest registers `http://127.0.0.1`.",
    )

    mail: bool = Field(default=True, description="Index the mailbox's messages.")

    calendar: bool = Field(default=True, description="Index the calendar's events.")

    exclude_folders: tuple[str, ...] = Field(
        default=(),
        max_length=256,
        description="Mail folder **paths** whose messages (and subfolders') are never indexed, "
        "written with slashes from the top: `Archive/2019`. Resolved to folder ids at the start "
        "of each run. Junk Email, Deleted Items, Drafts, Outbox and Sync Issues are always "
        "excluded and are not listed here. A path the mailbox does not have is a configuration "
        "error, never an empty exclusion.",
    )

    calendar_days_back: int = Field(
        default=90,
        ge=0,
        le=3_650,
        description="How far back from now the calendar window reaches.",
    )

    calendar_days_forward: int = Field(
        default=180,
        ge=0,
        le=3_650,
        description="How far ahead of now the calendar window reaches.",
    )

    max_concurrent_requests: int = Field(
        default=MAX_CONCURRENT_REQUESTS,
        ge=1,
        le=MAX_CONCURRENT_REQUESTS,
        description="Requests in flight at once. Outlook allows four per app and mailbox; the "
        "ceiling is that cap, because a request over it is throttled and not queued.",
    )

    max_retry_after_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Longest single pause honored from a `Retry-After` header; a longer request "
        "stops the run.",
    )

    max_retries: int = Field(
        default=5,
        ge=0,
        description="Attempts after the first for a throttled, failed or interrupted request.",
    )

    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("account")
    @classmethod
    def _account(cls, value: str) -> str:
        # Never echoed: it is the address this whole connector is careful not to print.
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
                "account must be of the form name@example.org: the sign-in name or the primary "
                "address of the Microsoft 365 account whose mailbox this instance indexes"
            )
            raise ValueError(msg)
        return address

    @field_validator("tenant_id")
    @classmethod
    def _tenant(cls, value: str) -> str:
        tenant = value.strip()
        if tenant.casefold() in REFUSED_TENANTS:
            msg = (
                f"tenant_id is {tenant!r}, an alias that lets whichever account signs in choose "
                f"the directory. An instance indexes one mailbox in one directory: use the "
                f"directory's GUID (Entra admin center, Overview, Tenant ID)"
            )
            raise ValueError(msg)
        if not _GUID.fullmatch(tenant):
            msg = (
                "tenant_id must be the directory's GUID (00000000-0000-0000-0000-000000000000). "
                "It becomes part of the token endpoint's address, so nothing else is accepted"
            )
            raise ValueError(msg)
        return tenant.casefold()

    @field_validator("client_id")
    @classmethod
    def _client(cls, value: str) -> str:
        client = value.strip()
        if not _GUID.fullmatch(client):
            msg = "client_id must be the app registration's application (client) id, a GUID"
            raise ValueError(msg)
        return client.casefold()

    @field_validator("exclude_folders")
    @classmethod
    def _exclude(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        paths: list[str] = []
        for raw in values:
            parts = [part.strip() for part in raw.split(_FOLDER_PATH_SEPARATOR)]
            if not raw.strip() or any(not part for part in parts):
                msg = (
                    "exclude_folders contains an empty path or an empty segment; remove it "
                    "rather than excluding nothing"
                )
                raise ValueError(msg)
            paths.append(_FOLDER_PATH_SEPARATOR.join(parts))
        return tuple(dict.fromkeys(paths))

    @model_validator(mode="after")
    def _checked(self) -> Self:
        if not (self.mail or self.calendar):
            msg = "mail and calendar are both false, so this instance would index nothing"
            raise ValueError(msg)
        if self.refresh_token is not None and not self.refresh_token.get_secret_value():
            msg = (
                f"refresh_token is set to an empty string. Leave it unset to read "
                f"${self.refresh_token_env}, or give it a value."
            )
            raise ValueError(msg)
        if self.calendar and self.calendar_days_back + self.calendar_days_forward < 1:
            msg = "the calendar window is empty: calendar_days_back and calendar_days_forward are 0"
            raise ValueError(msg)
        return self

    @property
    def pinned_identity(self) -> str:
        """A digest of the directory and mailbox, for the pipeline to compare across runs.

        A digest rather than either value, because it is stored beside the instance's run
        metadata, which status surfaces read, and those carry no address. The tenant is in it
        because the same address in another directory is another mailbox.
        """
        return hashlib.blake2b(
            f"outlook-account:{self.tenant_id}:{self.account}".encode(), digest_size=20
        ).hexdigest()

    @property
    def scopes(self) -> tuple[str, ...]:
        """The Graph scopes this instance asks for: read-only, and only what it indexes."""
        wanted = ["offline_access", "User.Read"]
        if self.mail:
            wanted.append("Mail.Read")
        if self.calendar:
            wanted.append("Calendars.Read")
        return tuple(wanted)

    def scope_fingerprint(self, source_scope: str, excluded_folder_ids: Iterable[str]) -> str:
        """The scope a stored set of delta links is valid for.

        Takes the **resolved** folder ids, because ids are what every discovery path tests and a
        renamed folder is the same scope. The representation versions and the two flags are in
        it so that changing what is fetched discards every stored position
        (``outlook.md`` §4.1). The calendar window is deliberately not: the calendar is listed
        afresh on every sync and keeps no position.
        """
        material = json.dumps(
            {
                "source_scope": source_scope,
                "account": self.account,
                "tenant": self.tenant_id,
                "excluded_folder_ids": sorted(set(excluded_folder_ids)),
                "mail": self.mail,
                "calendar": self.calendar,
                "mail_representation": MAIL_REPRESENTATION_VERSION,
                "event_representation": EVENT_REPRESENTATION_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()


def resolve_folder_ids(paths: Iterable[str], path_to_id: Mapping[str, str]) -> tuple[str, ...]:
    """The folder ids for configured folder paths, sorted and without duplicates.

    Pure, so the rule is testable without a mailbox. Matching is exact and case-insensitive on
    each segment, because Exchange folder names are.

    Raises:
        ConfigError: A path is not in ``path_to_id``. It names the **configured** path, which is
            the operator's own text, and never lists the folders the mailbox has: folder names
            are mail-derived and run-level errors carry none. Refused rather than skipped,
            because a typo that silently excluded nothing would index precisely the mail the
            operator meant to keep out.
    """
    by_key = {key.casefold(): value for key, value in path_to_id.items()}
    ids: set[str] = set()
    for path in paths:
        folder_id = by_key.get(path.casefold())
        if folder_id is None:
            msg = (
                f"exclude_folders names {path!r}, which is not a folder in this mailbox. Paths "
                f"are written from the top with slashes ('Archive/2019'), and a folder that does "
                f"not exist excludes nothing, which is why it is refused rather than ignored."
            )
            raise ConfigError(msg)
        ids.add(folder_id)
    return tuple(sorted(ids))


def read_token_file(path: str) -> SecretStr | None:
    """The refresh token in ``path``, or ``None`` when the file is absent or blank.

    Absent is not an error here: the file is where a rotated token is written, and a first run
    may name one that ``connector login`` has not yet created. The caller falls through to the
    environment, and the final refusal names every place that was tried.
    """
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        msg = (
            f"refresh_token_file {path!r} could not be read ({type(exc).__name__}). It should "
            f"hold the token `manicule connector login` wrote, and nothing else."
        )
        raise ConfigError(msg) from None
    return SecretStr(text) if text else None


def resolve_outlook_credentials(
    config: OutlookConfig, environ: Mapping[str, str] | None = None, *, instance: str = ""
) -> OutlookConfig:
    """Fill in the refresh token from its file or the environment, refusing any gap.

    Runs in the factory, before the connector is constructed, on the principle
    :func:`~manicule.connectors.config.resolve_credentials` states: a credential problem found at
    the first request of the first sync is a run that reports progress and indexes nothing. An
    explicitly configured value is never overwritten, the file wins over the environment (it is
    where the newer, rotated token is), and a blank value counts as unset.

    Raises:
        ConfigError: No refresh token anywhere. The message names every place, and never a value.
    """
    env = os.environ if environ is None else environ
    token = config.refresh_token
    if token is None and config.refresh_token_file is not None:
        token = read_token_file(config.refresh_token_file)
    if token is None:
        from_env = env.get(config.refresh_token_env)
        token = SecretStr(from_env) if from_env else None
    if token is None:
        where = (
            f"[connectors.{instance}.options]" if instance else 'plugins.config."connector.outlook"'
        )
        tried = [f"refresh_token in {where}"]
        if config.refresh_token_file is not None:
            tried.append("the file refresh_token_file names")
        tried.append(f"${config.refresh_token_env}")
        msg = (
            f"the Outlook connector needs a refresh token and found none in {', '.join(tried)}. "
            f"`manicule connector login {instance or '<name>'}` mints one."
        )
        raise ConfigError(msg)
    return config.model_copy(update={"refresh_token": token})
