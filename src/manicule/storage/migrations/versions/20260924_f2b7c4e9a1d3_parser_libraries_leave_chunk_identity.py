"""parser libraries leave the chunk fingerprint

Two parts of :class:`~manicule.core.fingerprints.ChunkFingerprint` described a parser rather
than the chunker: the ``grammars`` map, recording the tree-sitter grammar pack's release per
declared language, and the ``;html_text=web-blocks/1+selectolax/…`` suffix on ``version``,
recording the converter an HTML-only email body is reduced through. That fingerprint is compared
once per run for the whole corpus, so a bump of either library refused every ingest into every
existing index — 0.2.9's move of the grammar pack from 1.17.0 to 1.20.0 did exactly that, to
corpora holding no code at all. Both now live in the per-document ``ParseFingerprint`` of the
parsers that use them.

**This revision rewrites the stored fingerprints to the identity the code now computes**, in the
two places a chunk fingerprint is stored and compared as a string: ``documents.chunk_fp``, in
canonical form, and ``index_state.chunk_fingerprint``, as the model serialized it. Without it,
every existing document's ``chunk_fp`` would differ from the running chunker's by those two
fields alone, and everything that selects on that column — ``chunk_fp_other_than``, a rebuild's
plan — would report the whole corpus as built by some other chunker. The rewrite is exact rather
than optimistic: the chunks those rows describe *were* cut by the chunker the remaining fields
name. What the removed fields said about parsing is carried per document by ``parse_fp``, which
this revision does not touch; the code parser and ``msg`` now name libraries they did not before,
so their documents re-parse on the next sync, which is the correct price and the only one.

**Only rows that carry one of the two are touched**, and each is re-serialized in the shape it
was stored in — sorted keys and ASCII escapes for the canonical column, the model's own field
order for ``index_state`` — so a value that needed no change stays byte-for-byte what it was.

**The downgrade does nothing, and cannot.** The removed versions are not recoverable from the
rows that remain, and inventing them would assert a grammar or converter that nobody recorded.
An older build reading these rows sees an empty ``grammars`` map and no ``html_text`` suffix,
refuses the run against its own configuration, and says which fields differ.

Revision ID: f2b7c4e9a1d3
Revises: e41b9c07d2a5
Created: 2026-09-24 22:00:00.000000
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import sqlalchemy as sa
from alembic import op

revision: str = "f2b7c4e9a1d3"
down_revision: str | None = "e41b9c07d2a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RETIRED_FIELD = "grammars"
RETIRED_VERSION_COMPONENT = "html_text="


def retire(stored: str, *, canonical: bool) -> str | None:
    """``stored`` without the retired parts, or ``None`` when it carries neither.

    Args:
        stored: A serialized chunk fingerprint.
        canonical: Whether it is in :meth:`~manicule.core.fingerprints.Fingerprint.canonical`
            form, which decides how it is written back.

    Returns:
        The rewritten serialization, or ``None`` for a value this revision leaves alone — one
        that is not a JSON object, or that carries neither retired part.
    """
    try:
        value: object = json.loads(stored)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    fields = dict(cast("dict[str, object]", value))
    changed = RETIRED_FIELD in fields
    fields.pop(RETIRED_FIELD, None)
    version = fields.get("version")
    if isinstance(version, str):
        kept = [
            part
            for position, part in enumerate(version.split(";"))
            if position == 0 or not part.startswith(RETIRED_VERSION_COMPONENT)
        ]
        if len(kept) != version.count(";") + 1:
            fields["version"] = ";".join(kept)
            changed = True
    if not changed:
        return None
    if canonical:
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return json.dumps(fields, separators=(",", ":"), ensure_ascii=False)


def upgrade() -> None:
    connection = op.get_bind()
    distinct = connection.execute(
        sa.text("SELECT DISTINCT chunk_fp FROM documents WHERE chunk_fp IS NOT NULL")
    ).scalars()
    for stored in list(distinct):
        rewritten = retire(stored, canonical=True)
        if rewritten is not None:
            connection.execute(
                sa.text("UPDATE documents SET chunk_fp = :rewritten WHERE chunk_fp = :stored"),
                {"rewritten": rewritten, "stored": stored},
            )
    rows = connection.execute(
        sa.text(
            "SELECT workspace_id, chunk_fingerprint FROM index_state "
            "WHERE chunk_fingerprint IS NOT NULL"
        )
    ).all()
    for workspace, stored in rows:
        rewritten = retire(stored, canonical=False)
        if rewritten is not None:
            connection.execute(
                sa.text(
                    "UPDATE index_state SET chunk_fingerprint = :rewritten "
                    "WHERE workspace_id = :workspace"
                ),
                {"rewritten": rewritten, "workspace": workspace},
            )


def downgrade() -> None:
    # Nothing to restore: see the module docstring. The versions this revision removed are not
    # in the rows that remain, and writing guessed ones back would be a lineage nobody recorded.
    pass
