"""Filters that keep autogenerate away from objects Alembic cannot model.

In their own module rather than in ``env.py`` because ``env.py`` *runs* migrations when it is
imported — it is an Alembic script, not a library — so nothing can import it to test what it
declares. A filter that is never tested is a filter that quietly stops matching.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Literal

from manicule.storage.fts import FTS_SHADOW_TABLES

if TYPE_CHECKING:
    from sqlalchemy.sql.schema import SchemaItem

NameType = Literal[
    "schema",
    "table",
    "column",
    "index",
    "unique_constraint",
    "foreign_key_constraint",
    "check_constraint",
]
ParentNames = MutableMapping[
    Literal["schema_name", "table_name", "schema_qualified_table_name"], str | None
]

EXCLUDED_TABLES = frozenset(FTS_SHADOW_TABLES)
"""The FTS5 virtual table and its five shadow tables.

They exist in the database, are absent from the models, and are created by hand-written DDL.
Autogenerate does not model virtual tables, so without this filter every revision it produces
begins by dropping the lexical index.
"""


def include_name(name: str | None, type_: NameType, _parent_names: ParentNames) -> bool:
    """Whether autogenerate should consider an object it found in the database."""
    if type_ == "table" and name is not None:
        return name not in EXCLUDED_TABLES
    return True


def enum_check_constraints() -> frozenset[str]:
    """Names of the ``CHECK`` constraints that ``Enum`` columns generate.

    These exist in the database, because the migration created them, and they exist in the
    models, because the column type produces them. Autogenerate cannot match the two: the
    constraint is emitted during DDL generation rather than declared on the ``Table``, so the
    reflected one looks like a constraint the models no longer want and every comparison
    proposes dropping it.

    Computed from the metadata rather than hardcoded, so adding an enum column does not
    quietly reintroduce the false diff.
    """
    from sqlalchemy import Enum  # noqa: PLC0415 - keeps this module importable standalone

    from manicule.storage.models import Base  # noqa: PLC0415 - avoids an import cycle

    names: set[str] = set()
    for table in Base.metadata.sorted_tables:
        for column in table.columns:
            if isinstance(column.type, Enum) and column.type.name:
                names.add(f"ck_{table.name}_{column.type.name}")
    return frozenset(names)


def include_object(
    _obj: SchemaItem,
    name: str | None,
    type_: str,
    _reflected: bool,
    _compare_to: SchemaItem | None,
) -> bool:
    """The same exclusion, plus the enum constraints autogenerate cannot match."""
    if type_ == "table" and name is not None:
        return name not in EXCLUDED_TABLES
    if type_ == "check_constraint" and name is not None:
        return name not in enum_check_constraints()
    return True


__all__ = [
    "EXCLUDED_TABLES",
    "NameType",
    "ParentNames",
    "enum_check_constraints",
    "include_name",
    "include_object",
]
