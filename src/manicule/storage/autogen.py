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


UNREFLECTABLE_FOREIGN_KEYS = frozenset({("documents", ("container_id",))})
"""Foreign keys SQLite reports without their name, by table and constrained columns.

SQLite cannot ALTER a constraint onto an existing table, so a foreign key added after the
initial schema has to arrive inline in ``ADD COLUMN`` — the only form the engine accepts, and
the reason ``c7d1a4e83f26`` writes that one statement as raw DDL. The constraint is real:
``PRAGMA foreign_key_list`` reports it with ``ON DELETE CASCADE`` and the engine enforces it.
What is lost is the *name*, because SQLAlchemy recovers foreign-key names by parsing the
table-constraint form ``CONSTRAINT x FOREIGN KEY (...) REFERENCES ...`` out of the stored DDL,
and a column-level ``REFERENCES`` has no such clause to parse.

So autogenerate compares a reflected constraint with no name against a declared one the naming
convention named, concludes they are different constraints, and proposes dropping and adding —
on every fresh database, for ever. This is the same shape as :func:`enum_check_constraints`
above: present in the database, present in the models, and unmatchable between them.

Rebuilding ``documents`` would let the constraint be declared the reflectable way and is
exactly what none of these migrations may do — a batch rebuild fires every ``ON DELETE
CASCADE`` pointing at ``documents`` and empties the corpus's derived state while reporting
success.
"""


def _foreign_key_identity(obj: SchemaItem) -> tuple[str | None, tuple[str, ...]]:
    """A foreign key as ``(table, constrained columns)``, which is what both sides share.

    Matching on identity rather than on name is the point: the reflected side has no name, and
    a filter keyed on one would exclude the declared constraint and keep proposing to add it.
    """
    table = getattr(getattr(obj, "table", None), "name", None)
    columns = getattr(obj, "column_keys", None) or ()
    return table, tuple(str(column) for column in columns)


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
    if type_ == "foreign_key_constraint":
        return _foreign_key_identity(_obj) not in UNREFLECTABLE_FOREIGN_KEYS
    return True


__all__ = [
    "EXCLUDED_TABLES",
    "UNREFLECTABLE_FOREIGN_KEYS",
    "NameType",
    "ParentNames",
    "enum_check_constraints",
    "include_name",
    "include_object",
]
