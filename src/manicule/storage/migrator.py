"""Running migrations from application code and from tests.

Alembic is normally driven from a shell. Doing that from inside the process would mean
shelling out, or pointing Alembic at a URL while the application holds the same file open.
Both are worse than handing it the connection already in hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from manicule.core.errors import ManiculeError
from manicule.storage.engine import database_path, prepare_data_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(data_dir: Path | None = None) -> Config:
    """Build an Alembic config pointing at this package's migrations.

    Args:
        data_dir: When given, the database inside it becomes the target URL. Omit for
            operations that do not touch a database, such as reading the head revision.
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if data_dir is not None:
        config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database_path(data_dir)}")
    return config


def head_revision() -> str:
    """The revision the models are supposed to be at."""
    script = ScriptDirectory.from_config(alembic_config())
    head = script.get_current_head()
    if head is None:  # pragma: no cover - only reachable with an empty versions directory
        msg = "no Alembic head revision found; the versions directory is empty"
        raise RuntimeError(msg)
    return head


def current_revision(connection: Connection) -> str | None:
    """The revision a database is actually at, or ``None`` if it has never been migrated."""
    return MigrationContext.configure(connection).get_current_revision()


class StorageMigrationError(ManiculeError):
    """A migration ran and left the database in a state nothing should be written to.

    Distinct from a migration that *failed*, which rolls back and leaves a database somebody
    can retry against. This one is applied and inconsistent, and the honest instruction is to
    restore rather than to try again.
    """


def _upgrade(connection: Connection, revision: str) -> None:
    config = alembic_config()
    config.attributes["connection"] = connection
    command.upgrade(config, revision)


def _downgrade(connection: Connection, revision: str) -> None:
    config = alembic_config()
    config.attributes["connection"] = connection
    command.downgrade(config, revision)


async def upgrade(engine: AsyncEngine, revision: str = "head") -> None:
    """Migrate a database up, on a connection from this engine.

    Args:
        engine: The application's engine, so the pragmas in
            :func:`~manicule.storage.engine.attach_pragmas` apply to the migration too.
        revision: Target revision. ``"head"`` by default.
    """
    await _migrate(engine, _upgrade, revision)


async def downgrade(engine: AsyncEngine, revision: str) -> None:
    """Migrate a database down. Used by the round-trip check that keeps downgrades honest."""
    await _migrate(engine, _downgrade, revision)


async def _migrate(
    engine: AsyncEngine,
    run: Callable[[Connection, str], None],
    revision: str,
) -> None:
    """Run migrations with foreign keys off, and prove afterwards that they are still intact.

    **Off, deliberately, and only here.** SQLite cannot alter a constraint, so Alembic
    implements every such change as create-copy-``DROP``-rename — and with ``foreign_keys=ON``,
    which :func:`~manicule.storage.engine.attach_pragmas` sets on every connection, a
    ``DROP TABLE`` performs an implicit ``DELETE FROM`` first. That fires every
    ``ON DELETE CASCADE`` pointing at the table being rebuilt: rebuilding ``documents`` silently
    empties ``chunks``, ``document_versions``, ``document_tags`` and ``collection_documents``,
    and reports success. Where the cascade reaches ``chunks`` it also trips the FTS delete
    trigger, and the failed rebuild leaves a temporary table behind that makes every retry fail
    on a table that already exists.

    None of that is visible on an empty database, which is why it has to be stated here rather
    than discovered by the first person to migrate a real one.

    ``PRAGMA foreign_keys`` is a no-op inside a transaction, so it is set on the connection
    before one is opened. ``foreign_key_check`` then runs before the pragma goes back on: the
    safety net is disabled for the length of the migration and its absence is *checked*, rather
    than assumed to have been harmless.
    """
    async with engine.connect() as connection:
        # Each pragma is followed by a rollback. SQLAlchemy autobegins a transaction around any
        # statement, including these, and would then refuse the explicit ``begin()`` below;
        # pysqlite issues no DBAPI ``BEGIN`` for a pragma, so there is nothing to undo and the
        # rollback only clears the bookkeeping. It matters that no transaction is open when the
        # pragma runs — SQLite makes ``foreign_keys`` a silent no-op inside one, which would
        # leave the enforcement on and the cascade armed.
        await connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        await connection.rollback()
        try:
            async with connection.begin():
                await connection.run_sync(run, revision)
            violations = (await connection.exec_driver_sql("PRAGMA foreign_key_check")).all()
            await connection.rollback()
        finally:
            await connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            await connection.rollback()
        if violations:
            msg = (
                f"migrating to {revision!r} left {len(violations)} dangling foreign key "
                f"reference(s), the first being {violations[0]}. The schema change is applied "
                f"and the data is not consistent with it; restore from a backup."
            )
            raise StorageMigrationError(msg)


async def current(engine: AsyncEngine) -> str | None:
    """The revision this database is at."""
    async with engine.connect() as connection:
        return await connection.run_sync(current_revision)


async def initialise(data_dir: Path) -> None:
    """Create the data directory and bring its database to head.

    The one call an application needs at startup.
    """
    from manicule.storage.engine import create_engine  # noqa: PLC0415 - avoids a cycle

    prepare_data_dir(data_dir)
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
    finally:
        await engine.dispose()


__all__ = [
    "MIGRATIONS_DIR",
    "alembic_config",
    "current",
    "current_revision",
    "downgrade",
    "head_revision",
    "initialise",
    "upgrade",
]
