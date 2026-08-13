"""re-key documents on the identity their source declares

A local file's identity used to be its resolved path, always. It is now the ``source_id`` a
sidecar manifest declares, where one does, so that a mirror reorganised from by-space to by-tree
updates its pages instead of replacing every one of them. This revision moves the documents that
were already ingested, so that the next sync updates them rather than indexing a second copy
beside the first.

**Why a migration here, when the connector-instance rename in #94 shipped without one.** That
change had nothing in the database recording which instance a row came from, so a migration would
have had to guess; and its orphans were content a re-sync rebuilds identically. Neither is true
here. The mapping is stored data — ``documents.source_id`` holds the old identity and the
provenance record's own ``source_id`` holds the new one, written by the same fetch, for exactly
the rows that move — so the affected set is a query rather than an estimate. And a re-sync would
be **lossy**: ``collection_documents`` and ``document_tags`` hang off ``documents.id``, they hold
what a person assigned by hand, and nothing rebuilds them. Meanwhile the old rows stay live and
searchable, because nothing in the product calls reconciliation.

**Updated in place. Never inserted and deleted, and that is the whole design.** ``documents.id``
is the parent of five ``ON DELETE CASCADE`` foreign keys — ``chunks``, ``document_versions``,
``glossary_entries``, ``collection_documents``, ``document_tags``. Writing the new row and
dropping the old would fire every one of them and destroy the curation this exists to preserve,
which is this project's worst near-miss exactly: a migration that cascade-deleted four tables
while its round-trip test ran against an empty database, green and catastrophic. So the parent's
key is updated and each child's reference is updated with it, inside one transaction with
``PRAGMA defer_foreign_keys`` on, so enforcement happens once at commit rather than between two
statements that are individually inconsistent.

**Chunks and vectors are deliberately left where they are.** They are the generic HTML parse of
an enriched wrapper, so they are stale text — but the next sync replaces them, because
``content_hash`` now digests the extracted body and the adapter version is part of the change
token. Deleting them here would remove content before its replacement exists, which is a worse
intermediate state than serving one more sync's worth of the old parse. ``chunks.id`` does not
move, so the vectors keyed on it stay valid throughout.

**Nothing is deleted, and nothing that cannot be re-keyed is touched.** A row whose provenance
yields no page id is not affected and is left exactly as it was. A row whose new identity is
already taken — by a page ingested under its page id from somewhere else, or by a second document
declaring the same id — is skipped and logged, because re-keying onto an occupied primary key
means overwriting a document this revision cannot restore. The ``document-identity`` check in
``doctor`` keeps naming those until a person resolves them.

**The previous identity is recorded before it is overwritten**, under
:data:`PREVIOUS_IDENTITY` in ``documents.metadata``. Without it the downgrade would be a
fiction: the old identity was an absolute path, and once ``source_id`` holds the page id nothing
in the database remembers what the path was — the snapshot's location is recorded *relative to an
ingestion root* the database does not store. A downgrade that could not restore what it undid
would be untested code needed exactly once, under pressure.

**``document_id`` is imported rather than reimplemented, which is the opposite of the usual rule
for a migration and is deliberate.** The new key has to be the one the running connector will
look the document up by. A copy of the derivation frozen into this file would drift the first
time the real one changed, and the drift is silent: every document re-keyed to an id nothing ever
queries, present, indexed, and unreachable.

Revision ID: b2e6d0c94a17
Revises: 5f1c8a34b7d9
Created: 2026-08-13 09:00:00.000000
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, Final, cast

import sqlalchemy as sa
from alembic import op

from manicule.connectors.enriched import ENRICHED_KEY, AdapterOutcome
from manicule.core.ids import document_id
from manicule.core.provenance import PROVENANCE_KEY

revision: str = "b2e6d0c94a17"
down_revision: str | None = "5f1c8a34b7d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_log = logging.getLogger("alembic.runtime.migration")

IDENTITY_NOT_APPLIED: Final = AdapterOutcome.IDENTITY_NOT_APPLIED.value
"""The outcome marking a document this revision must leave where it is.

Imported rather than spelled, so that renaming the outcome breaks this revision loudly instead of
quietly widening what it moves."""

PREVIOUS_IDENTITY: Final = "previous_identity"
"""``documents.metadata`` key recording what a re-keyed document used to be called.

Written on the way up so the way down is real, and kept afterwards because "this document used to
be keyed on that path" is a fact an audit of a moved corpus needs and nothing else records.
"""

_CHILDREN: Final[tuple[str, ...]] = (
    "chunks",
    "document_versions",
    "glossary_entries",
    "collection_documents",
    "document_tags",
)
"""Every table with a ``document_id`` referencing ``documents.id``.

Written out rather than discovered from the metadata at runtime, so that a table added later
without a line here fails this revision's test rather than silently losing its rows — and so a
reader can see the whole blast radius of the key change in one place. ``glossary_aliases`` is
absent because it hangs off ``glossary_entries`` and never names a document.
"""


def _selection(direction: str) -> sa.TextClause:
    """The documents this revision moves, in one query rather than a scan.

    ``json_valid`` guards ``json_extract``, which raises on a column that is not JSON. That is not
    defensive noise: ``metadata`` is written by the application and read here by a different
    process at a different version, and a migration that raises partway through leaves a database
    half-moved.
    """
    if direction == "up":
        declared = f"json_extract(metadata, '$.{PROVENANCE_KEY}.source.source_id')"
    else:
        declared = f"json_extract(metadata, '$.{PREVIOUS_IDENTITY}.source_id')"
    # **Excluded: a page whose identity is available and deliberately not applied.** An enriched
    # export with no manifest beside it is adapted and cited by its own page id, and is keyed on
    # its path on purpose — because identity has to be known at *discovery*, and discovery reads
    # the manifest rather than the document. Re-keying one would move the row onto an identity the
    # connector will never look it up by: the next sync discovers the file under its path, finds
    # nothing, and creates a second document. The migration would cause exactly the duplication it
    # exists to prevent. Found by running an ingest and a migration over a real corpus, not by
    # anything failing.
    applied = f"json_extract(metadata, '$.{ENRICHED_KEY}.outcome')"
    return sa.text(
        f"SELECT id, workspace_id, source, source_id, metadata, {declared} AS declared "  # noqa: S608
        f"FROM documents "
        f"WHERE json_valid(metadata) "
        f"AND {declared} IS NOT NULL AND {declared} != '' AND {declared} != source_id "
        f"AND {applied} IS NOT '{IDENTITY_NOT_APPLIED}' "
        f"ORDER BY id"
    )


def _recorded(held: dict[str, Any]) -> str:
    """The identity :data:`PREVIOUS_IDENTITY` recorded, or ``""`` when it recorded none."""
    previous: object = held.get(PREVIOUS_IDENTITY)
    if not isinstance(previous, dict):
        return ""
    recorded: object = cast("dict[str, object]", previous).get("document_id")
    return str(recorded or "")


def _rekey(connection: sa.Connection, rows: Sequence[Any], *, direction: str) -> None:
    """Move each document onto its new key, taking its children with it.

    Raises nothing for a document it cannot move. A collision is reported and skipped, because
    the alternative is overwriting a row this revision has no way to put back.
    """
    # Deferred rather than disabled. Enforcement still happens — once, at commit — so a child
    # left pointing at a key that no longer exists is a hard failure rather than a corruption
    # nobody notices. Disabling the pragma outright would make the migration unable to fail.
    connection.execute(sa.text("PRAGMA defer_foreign_keys = ON"))
    taken = {row[0] for row in connection.execute(sa.text("SELECT id FROM documents")).fetchall()}
    for row in rows:
        old_id, workspace, source, old_source_id, metadata, declared = row
        held: dict[str, Any] = json.loads(metadata) if metadata else {}
        # **Up derives the key; down reads the one that was recorded.** They are not the same
        # operation reversed. Deriving on the way down assumes the old id *was* a derivation of
        # the old source id, and a row predating some earlier change need not be — so a
        # re-derived downgrade would put the document back under an id it never had, with its
        # children pointing at it and nothing anywhere saying so. Recording the previous identity
        # is what makes the reverse exact rather than plausible.
        new_id = document_id(workspace, source, declared) if direction == "up" else _recorded(held)
        if not new_id:  # pragma: no cover - the selection requires the key this reads
            continue
        if new_id in taken and new_id != old_id:
            _log.warning(
                "document %s declares source_id %r, whose identity %s is already taken by another "
                "document. Left keyed on %r; `manicule doctor` reports it and a person decides "
                "which page owns the identity.",
                old_id,
                declared,
                new_id,
                old_source_id,
            )
            continue
        if direction == "up":
            held[PREVIOUS_IDENTITY] = {"source_id": old_source_id, "document_id": old_id}
        else:
            held.pop(PREVIOUS_IDENTITY, None)
        for table in _CHILDREN:
            connection.execute(
                sa.text(
                    # S608: `table` comes from the literal `_CHILDREN` tuple above and a table
                    # name cannot be a bind parameter. Both values are bound.
                    f"UPDATE {table} SET document_id = :new WHERE document_id = :old"  # noqa: S608
                ),
                {"new": new_id, "old": old_id},
            )
        connection.execute(
            sa.text(
                "UPDATE documents SET id = :new, source_id = :source_id, metadata = :metadata "
                "WHERE id = :old"
            ),
            {
                "new": new_id,
                "source_id": declared,
                "metadata": json.dumps(held),
                "old": old_id,
            },
        )
        taken.discard(old_id)
        taken.add(new_id)


def upgrade() -> None:
    connection = op.get_bind()
    _rekey(connection, connection.execute(_selection("up")).fetchall(), direction="up")


def downgrade() -> None:
    """Put every document this revision moved back where it was.

    Exact rather than approximate, because :data:`PREVIOUS_IDENTITY` recorded the identity before
    it was overwritten. A document that was never moved carries no such key and is not selected,
    so a downgrade over a corpus this revision found nothing to do in does nothing at all.
    """
    connection = op.get_bind()
    _rekey(connection, connection.execute(_selection("down")).fetchall(), direction="down")
