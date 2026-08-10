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

from manicule.storage.engine import database_path, prepare_data_dir

if TYPE_CHECKING:
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
    async with engine.begin() as connection:
        await connection.run_sync(_upgrade, revision)


async def downgrade(engine: AsyncEngine, revision: str) -> None:
    """Migrate a database down. Used by the round-trip check that keeps downgrades honest."""
    async with engine.begin() as connection:
        await connection.run_sync(_downgrade, revision)


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
