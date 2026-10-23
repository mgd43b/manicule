"""Which sources make an installation too private to serve without authentication.

The index is not permission-aware: whoever can search a workspace reads everything in it. For
most sources that is a decision about who may use the installation. For a mailbox it is not a
decision anyone made — one person's mail would be readable by every process that can reach the
port — so a connector type registers itself as **sensitive**
(:attr:`~manicule.plugins.registry.ComponentRecord.sensitive`, beside its ``config_model``) and
:func:`~manicule.app.bind.require_sensitive_source_authentication` refuses to serve such an
installation without authentication, on every address.

**Two facts make an installation sensitive, and the second exists because the first is not
enough.**

1. *An enabled connector instance is of a sensitive type.* Read from configuration, so it is
   true before the first message is fetched.
2. *The data directory holds a document, in any state, from a source a sensitive connector has
   synced.* Configuration alone was the first proposal and was rejected: it stops protecting
   the mail at the moment somebody disables the connector or deletes it from the file, and the
   mail is still in the index. The pipeline records a mark on the source's ``connectors`` row
   before its first document is written (:meth:`~manicule.ingest.pipeline.IngestPipeline`
   does this through ``mark_source_sensitive``), the mark is never cleared, and this module asks
   whether any ``documents`` row still carries a marked source's name. Soft-deleted and purged
   rows count — a purged row keeps its title, which for mail is the subject — so only deleting
   the source's documents outright (``manicule document delete-source``) lifts the refusal.

**Read with plain ``sqlite3``, read-only, and not through the document store.** The refusal is
decided by :func:`~manicule.app.bind.resolve_bind` and
:func:`~manicule.api.app.build_app`, which are synchronous and run before the server's event
loop is serving, and by the ``doctor`` ``transport`` check, which must answer without opening a
writer. The document store is SQLite only, and the same read-only open is how
:mod:`manicule.storage.backup` copies a live database.

**Every workspace in the data directory is read, not only this process's.** The data directory
is the unit that holds mail, and a process serving one workspace is still serving a data
directory whose other workspaces' files sit beside it.

**A database that cannot be read refuses; it does not pass.** Only the two answers that prove
there is nothing to find are treated as nothing: no database file, and a database from before
the mark existed. Anything else — a lock, a corrupt file — is a refusal naming the cause,
because the alternative is serving possibly-private mail on the strength of an error.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError
from manicule.storage.engine import database_path

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from pathlib import Path

    from manicule.config.settings import Settings

_READ_TIMEOUT_S = 5.0
"""How long to wait on a writer's lock before refusing. Short: this runs at every start."""

_HELD = (
    "SELECT DISTINCT c.name FROM connectors AS c "
    "WHERE c.sensitive_synced_at IS NOT NULL "
    "AND EXISTS (SELECT 1 FROM documents AS d "
    "WHERE d.workspace_id = c.workspace_id AND d.source = c.name) "
    "ORDER BY c.name"
)
"""Marked sources with at least one document row of any state. No ``deleted_at`` filter."""

_PRE_MARK = ("no such table", "no such column")
"""What SQLite says about a database that predates the mark: it cannot hold one."""


@dataclass(frozen=True, slots=True)
class SensitiveExposure:
    """What makes an installation sensitive, by source (connector instance) name."""

    configured: tuple[str, ...] = ()
    """Enabled connector instances whose registered type is sensitive."""
    held: tuple[str, ...] = ()
    """Sources a sensitive connector has synced that still have a document row."""

    @property
    def sources(self) -> tuple[str, ...]:
        """Every name, once, sorted."""
        return tuple(sorted({*self.configured, *self.held}))

    def __bool__(self) -> bool:
        return bool(self.configured or self.held)


def sensitive_instances(settings: Settings, sensitive_types: AbstractSet[str]) -> frozenset[str]:
    """Every configured instance of a sensitive type, enabled or not.

    What the pipeline marks on sync. Disabled instances are included, since a manual sync of a
    disabled instance is still a sync.
    """
    return frozenset(
        name
        for name, configured in settings.connectors.items()
        if configured.type in sensitive_types
    )


def sensitive_exposure(settings: Settings, sensitive_types: AbstractSet[str]) -> SensitiveExposure:
    """What makes this installation sensitive, from configuration and from the data directory.

    Args:
        settings: Supplies the configured instances and the data directory.
        sensitive_types: The connector types the registry marks sensitive
            (:meth:`~manicule.plugins.registry.ComponentRegistry.sensitive_names`).

    Raises:
        PolicyError: The data directory's database exists and could not be read.
    """
    configured = tuple(
        sorted(
            name
            for name, instance in settings.connectors.items()
            if instance.enabled and instance.type in sensitive_types
        )
    )
    return SensitiveExposure(configured=configured, held=held_sources(settings.data_dir))


def held_sources(data_dir: Path) -> tuple[str, ...]:
    """Marked sources whose documents are still in ``data_dir``'s database, in any state."""
    path = database_path(data_dir)
    if not path.exists():
        return ()
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_READ_TIMEOUT_S)
    except sqlite3.Error as error:
        raise _unreadable(path, error) from error
    try:
        return tuple(name for (name,) in connection.execute(_HELD))
    except sqlite3.OperationalError as error:
        if any(fragment in str(error) for fragment in _PRE_MARK):
            return ()
        raise _unreadable(path, error) from error
    except sqlite3.Error as error:
        raise _unreadable(path, error) from error
    finally:
        connection.close()


def exposure_when_unauthenticated(
    settings: Settings, sensitive_types: AbstractSet[str]
) -> SensitiveExposure:
    """:func:`sensitive_exposure`, or nothing when authentication is on.

    The data directory is read only when the answer can matter, so an installation that
    authenticates is never refused because its database could not be read.
    """
    if settings.security.auth.mode is not AuthMode.NONE:
        return SensitiveExposure()
    return sensitive_exposure(settings, sensitive_types)


def _unreadable(path: Path, error: sqlite3.Error) -> PolicyError:
    msg = (
        f"refusing to serve without authentication: the database at {path} could not be read "
        f"to tell whether it holds documents from a sensitive source ({error}). Set "
        f"security.auth.mode to 'api_key' or 'oauth', or fix the database."
    )
    return PolicyError(msg)


__all__ = [
    "SensitiveExposure",
    "exposure_when_unauthenticated",
    "held_sources",
    "sensitive_exposure",
    "sensitive_instances",
]
