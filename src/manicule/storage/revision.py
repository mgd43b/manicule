"""The schema revision this code expects, and the one a database is at, without Alembic.

Every process that opens the index asks whether its schema is behind, and almost always the
answer is no. Asking Alembic meant importing it and reading every script in
``migrations/versions`` to work out which one is the head — about fifty milliseconds a command,
on a command line where the work asked for often takes one. The head is a fact about this
release, so it is written down here; ``tests/test_storage_migrations.py`` fails the moment a new
migration lands without it being updated, so it cannot drift. Alembic is imported only on the
path that actually migrates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

HEAD_REVISION: Final = "f4b1d8a27c63"
"""The newest migration in ``migrations/versions``. Update it with every new migration."""


def current_revision(connection: Connection) -> str | None:
    """The revision a database records, ``None`` if it has never been migrated.

    Read from Alembic's own table with one statement. A database recording more than one head
    is not one this code's linear history produces, so it reports a value that is never
    :data:`HEAD_REVISION`, and the caller takes the migrating path, where Alembic says what is
    wrong in its own words.
    """
    # Asked of the catalog rather than inferred from a failed SELECT. Catching the failure would
    # read a locked, damaged or unreadable database as one that was never migrated, and send a
    # reader into the lock-and-upgrade path against it instead of reporting what went wrong.
    present = connection.execute(
        text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alembic_version'")
    ).first()
    if present is None:
        return None
    rows = connection.execute(text("SELECT version_num FROM alembic_version")).all()
    if not rows:
        return None
    if len(rows) > 1:
        return ",".join(sorted(str(row[0]) for row in rows))
    return str(rows[0][0])


__all__ = ["HEAD_REVISION", "current_revision"]
