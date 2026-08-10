"""Alembic environment.

Four things here are load-bearing, and three of them are specific to SQLite.

``render_as_batch``
    SQLite cannot drop a column or alter a constraint. Batch mode implements both as
    create-copy-swap. Without it the first migration that changes a ``CHECK`` cannot be
    written at all.

The naming convention
    Comes from :data:`manicule.storage.models.Base`. Batch mode has to *name* the constraint
    it drops, and SQLite generates anonymous ones.

The autogenerate filters
    Imported from :mod:`manicule.storage.autogen` rather than defined here, because this
    module runs migrations when imported and so cannot be imported by a test.

Async
    The application engine is ``aiosqlite``; migrations run through ``run_sync`` so that one
    URL works for both.
"""

from __future__ import annotations

import asyncio
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from manicule.storage.autogen import include_name, include_object
from manicule.storage.models import Base

config = context.config
target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
        include_name=include_name,
        include_object=include_object,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a database, for review."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        include_name=include_name,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    section: dict[str, Any] = config.get_section(config.config_ini_section) or {}
    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations against a live database.

    Accepts an already-open connection through ``config.attributes`` so that application code
    and tests can migrate on a connection they control, rather than having Alembic open a
    second one against the same file.
    """
    existing = config.attributes.get("connection")
    if isinstance(existing, Connection):
        _do_run_migrations(existing)
        return
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
