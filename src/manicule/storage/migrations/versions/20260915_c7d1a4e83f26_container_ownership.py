"""persisted ownership of container-derived documents

``documents`` gains ``container_id`` and ``container_depth``. Until now a member of an archive
or a mail message was a document with no recorded parent: the relationship existed only inside
its ``source_id``, as the ``zip:<container>!/<path>`` spelling, which nothing joined on and
nothing enforced.

**Two defects followed from that, and they pull in opposite directions.** Connector-level
reconciliation diffs the source ids a connector reports against the ids this table holds, and a
member's id was never reported by any connector and never could be — so every member of every
container was permanently in the missing set, and a pass that got past the deletion ceiling would
soft-delete documents that were perfectly current. Meanwhile a member that really had been
removed from its archive stayed indexed forever, because nothing reconciled a container's
children against what it now expands to. Deleting valid members and keeping invalid ones, from
the same absent column.

**Existing rows are adopted rather than left ``NULL``**, which is the opposite of the choice
``a3f7c1d8e520`` made for ``relation_fp``, and the difference is worth stating. A ``NULL``
lineage fingerprint is *true*: nothing had derived those edges. A ``NULL`` container id on a
document that plainly is a member would be *false*, and it would be false in the direction that
keeps the deletion hazard alive until every container happens to re-sync. So ownership is
recovered from the identity that already encodes it: a member's ``source_id`` is its scheme, its
parent's ``source_id``, ``!/``, and the path inside. Each candidate parent is resolved against a
document that actually exists before it is written, so a source id that merely looks like one of
these adopts nothing.

Depth is then filled by walking the ownership that was just recorded, bounded by the nesting
ceiling in ``manicule.parsers.expansion`` so that a cycle — which the expansion path refuses but
a hand-edited database could hold — costs a fixed number of passes rather than a hang.

**Plain ``ADD COLUMN``s, deliberately not a batch rebuild**, for the reason ``a71f3c9d0e55`` sets
out and ``a3f7c1d8e520`` repeats: SQLite implements a batch alteration as CREATE-temp,
INSERT…SELECT, ``DROP TABLE``, RENAME, and with ``PRAGMA foreign_keys=ON`` — which this project
sets on every connection — that ``DROP`` fires every ``ON DELETE CASCADE`` pointing at
``documents``, emptying chunks, versions, tags, collection membership and glossary entries while
reporting success. That is also why ``container_depth`` arrives with a server default and keeps
it: dropping the default afterwards is precisely the rebuild this cannot do, so the model
declares the same default rather than a second pass removing it.

Revision ID: c7d1a4e83f26
Revises: b5e2c73a91d4
Created: 2026-09-15 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7d1a4e83f26"
down_revision: str | None = "b5e2c73a91d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SEPARATOR = "!/"
"""``manicule.parsers.expansion.CONTAINER_SEPARATOR``, spelled out rather than imported.

A migration describes the schema at one revision, and importing a constant would make this
revision's behavior change whenever that constant does — which is the one thing a migration
already applied must never do.
"""

_SCHEMES = ("zip", "mail")
"""The member schemes in use at this revision: ``ARCHIVE_SCHEME`` and ``MAIL_SCHEME``."""

_MAX_DEPTH = 3
"""``manicule.parsers.expansion.MAX_DEPTH``, and here a bound on the settling passes below."""


def upgrade() -> None:
    # Raw DDL for this one column, and it is not a shortcut. `op.add_column` with a
    # `ForeignKey` asks the dialect to add the column and then ALTER the constraint on, which
    # SQLite has no support for — it answers `NotImplementedError` and points at batch mode,
    # which is the copy-and-move rebuild of `documents` that the docstring above forbids
    # outright. SQLite does accept a REFERENCES clause *inline* in ADD COLUMN, provided the
    # new column defaults to NULL, which this one does. The constraint is named by hand to the
    # metadata naming convention, because an unnamed one reads back as drift against the model
    # and `alembic check` — which this project runs as a test — would fail on every fresh
    # database.
    op.execute(
        sa.text(
            "ALTER TABLE documents ADD COLUMN container_id TEXT "
            "CONSTRAINT fk_documents_container_id_documents "
            "REFERENCES documents (id) ON DELETE CASCADE"
        )
    )
    op.add_column(
        "documents",
        sa.Column("container_depth", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_index("ix_documents_container_id", "documents", ["container_id"], unique=False)
    _adopt_existing_members()


def downgrade() -> None:
    # Index first: SQLite's ALTER TABLE DROP COLUMN refuses a column an index refers to.
    op.drop_index("ix_documents_container_id", table_name="documents")
    op.drop_column("documents", "container_depth")
    op.drop_column("documents", "container_id")


def _adopt_existing_members() -> None:
    """Record the ownership that a member's ``source_id`` has been encoding all along."""
    bind = op.get_bind()
    documents = sa.table(
        "documents",
        sa.column("id", sa.Text),
        sa.column("workspace_id", sa.Text),
        sa.column("source", sa.Text),
        sa.column("source_id", sa.Text),
        sa.column("container_id", sa.Text),
        sa.column("container_depth", sa.Integer),
    )
    rows = bind.execute(
        sa.select(
            documents.c.id, documents.c.workspace_id, documents.c.source, documents.c.source_id
        )
    ).all()
    by_identity = {(row.workspace_id, row.source, row.source_id): row.id for row in rows}

    owners: dict[str, str] = {}
    for row in rows:
        parent = _parent_of(row.source_id, row.workspace_id, row.source, by_identity)
        if parent is not None:
            owners[row.id] = parent
    for child, parent in owners.items():
        bind.execute(
            sa.update(documents).where(documents.c.id == child).values(container_id=parent)
        )

    # Depth settles upward: a child of a top-level document reaches 1 on the first pass, a child
    # of that child on the second. Bounded by the nesting ceiling rather than run to a fixed
    # point, so a cycle this schema cannot create but a hand-edited database could hold costs
    # three passes instead of hanging the upgrade.
    parents = documents.alias("parents")
    for _ in range(_MAX_DEPTH):
        bind.execute(
            sa.update(documents)
            .where(documents.c.container_id.is_not(None))
            .values(
                container_depth=sa.select(parents.c.container_depth + 1)
                .where(parents.c.id == documents.c.container_id)
                .scalar_subquery()
            )
        )


def _parent_of(
    source_id: str, workspace_id: str, source: str, by_identity: dict[tuple[str, str, str], str]
) -> str | None:
    """The document id this member was expanded out of, or ``None`` if it is not one.

    A member's identity is ``<scheme>:<parent source_id>!/<path inside>``, and a path inside may
    itself contain the separator, so candidates are tried longest-first and the first one that
    names a document that exists wins. Resolving against the table rather than trusting the
    spelling is what stops a connector whose own ids happen to contain ``!/`` from adopting
    documents that are not its members.
    """
    for scheme in _SCHEMES:
        prefix = f"{scheme}:"
        if not source_id.startswith(prefix):
            continue
        rest = source_id[len(prefix) :]
        cut = rest.rfind(_SEPARATOR)
        while cut > 0:
            candidate = by_identity.get((workspace_id, source, rest[:cut]))
            if candidate is not None and candidate != by_identity.get(
                (workspace_id, source, source_id)
            ):
                return candidate
            cut = rest.rfind(_SEPARATOR, 0, cut)
    return None
