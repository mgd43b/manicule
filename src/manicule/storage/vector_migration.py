"""Carry one workspace's vectors from one backend into another, without re-embedding them.

**Why this exists rather than a re-index.** SQLite holds ``chunks.embed_text`` — exactly what
the embedder saw — but it holds no vectors, and there is no persistent embedding cache. So
pointing ``storage.vector_db`` at an empty store makes every chunk read ``ABSENT`` on the reuse
path, and the next ingest is a full forward pass of the embedder over the corpus.
``docs/storage.md`` §9.4 says so plainly for the restore case and offers two postures, neither
of which is "move the index you already have". This is that third posture.

**What makes it a copy rather than a conversion, and what makes it backend-agnostic.** Every
backend stores the same row. A Lance column, a Qdrant payload field and a relational column
carry the same names, from the same object, under the same rules, and
:mod:`manicule.storage.vector_schema` is where that shape is written down once
(``docs/storage.md`` §6.7). So a migration is a read in one shape and a write in the same shape,
and this module names no backend at all: it asks one store for
:class:`~manicule.core.protocols.InspectableVectorStore` and the other for
:class:`~manicule.core.protocols.AdoptingVectorStore`. A backend added later becomes migratable
by implementing a capability rather than by being added to a list here — which is the same rule
``docs/storage.md`` §6.7 states about asking a capability of the object rather than of the
module that would implement it.

**The two halves are separate protocols because the backends are not symmetric.** The embedded
store is the one installations start on, so it is the one they read *from*; a networked store is
the one they move *to*. Requiring both halves of every backend would make each of them claim a
capability in order to offer the other.

**No embedder is built, and that is load-bearing.** The fingerprint comes from what the source
itself recorded beside its vectors. Asking the configured embedder instead would make moving an
index impossible on a machine whose model runtime is broken or whose weights are not present —
which is a machine that still deserves a working index, and is disproportionately the machine
somebody is migrating *off*. It is the same argument
:meth:`~manicule.app.runtime.Runtime.vectors` makes for reporting checksum coverage through the
unprepared handle.

**What this establishes, and the two things it does not.** It establishes that every row the
source held was written to the destination with its numbers unchanged and its recorded checksum
still describing them — a claim about a backend rather than about arithmetic, since a store whose
distance is cosine may re-normalize on write, and ``docs/storage.md`` §6.8 records which ones
measurably do. It does not establish that the source held everything the corpus expects
— that is a count against ``chunks.vector_id``, which the caller reports beside this — and it
does not establish that the destination will keep them, which is the destination's durability
and not this project's (``docs/storage.md`` §9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, NoReturn

from manicule.core.embedding import VectorIntegrity
from manicule.core.errors import VectorMigrationError
from manicule.storage.vector_schema import CHUNK_ID_COLUMN, ID_COLUMN, row_integrity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from manicule.core.protocols import AdoptingVectorStore, InspectableVectorStore

MIGRATION_PAGE: Final = 256
"""Rows read from the source and written to the destination at a time.

Matches the embedded store's own default inspection page. The destination batches again
internally at its own upsert bound, so this is the memory bound rather than the request size:
one page of rows, each carrying a vector and a serialized chunk, is what is resident at once.
"""

PROGRESS_ROWS: Final = 5_000
"""How many rows pass between progress sentences.

A line per page would be a line per 256 rows, which on a corpus large enough for the migration
to be worth doing is thousands of lines of scrollback that say nothing a running total does not.
"""


@dataclass(frozen=True, slots=True)
class VectorMigration:
    """What one migration moved, or what a plan says one would move."""

    storage_name: str
    """What the destination calls the place the vectors landed, so an operator can look."""

    generation: str
    """The source generation — ``legacy``, or the ``reembed-…`` pointer that supersedes it."""

    dimension: int
    source_rows: int
    copied: int
    unverified: int
    """Rows carried that record no checksum.

    Written before the numerical-integrity contract existed, so there was nothing to verify
    rather than something that failed. They are carried with the absence intact, because
    manufacturing a checksum for them here would assert about the embedder's output something
    only the embedder ever knew.
    """

    dry_run: bool


async def migrate_vectors(
    source: InspectableVectorStore,
    target: AdoptingVectorStore,
    *,
    generation: str,
    report: Callable[[str], None] | None = None,
    dry_run: bool = True,
    page_size: int = MIGRATION_PAGE,
) -> VectorMigration:
    """Copy every row the source generation holds into the destination collection.

    Args:
        source: A store already opened on what is being moved. Opened by the caller, because
            finding a generation is backend-specific — a directory, a DSN, a collection — while
            reading one is not. What this requires of it is only that it can say which model
            its vectors came from and hand them back in the shared row shape.
        target: The configured destination. Prepared here, which is what creates its
            storage and whatever indexes it wants if this is the first time.
        generation: What the source directory stands for — ``legacy`` for the published root,
            or the ``reembed-…`` pointer a durable re-embed swapped in. Passed in rather than
            read off the store, because the caller resolved the pointer to find the directory
            and asking the store to name it again would be a second answer to one question.
        report: Progress sentences, at :data:`PROGRESS_ROWS` intervals. ``None`` when
            nobody is listening, which is every in-process call.
        dry_run: Count what would move and write nothing. The default.
        page_size: Rows resident at once.

    Returns:
        What moved, or what would move.

    Raises:
        VectorMigrationError: The source records no fingerprint, the destination already holds
            rows, or a source row's recorded checksum no longer describes its vector.
        FingerprintMismatchError: The destination was prepared for a different model.
    """
    notify = report if report is not None else _quiet
    fingerprint = await source.fingerprint()
    if fingerprint is None:
        _refuse_unbuilt()
    # Everything a plan needs, and nothing a plan should cause. `storage_name` is arithmetic,
    # `fingerprint` and `count` read what the destination already recorded, and none of the
    # three creates storage — so `--dry-run` against a destination that has never been written
    # to leaves it exactly as it found it, rather than leaving an empty collection behind for
    # an operator who then decided not to migrate.
    # Asked of the space this migration is about, not of whatever the destination has recorded
    # about itself: a destination whose identity record is missing while its storage is not
    # would otherwise answer zero to the one question standing between this and a merge.
    recorded = await target.fingerprint()
    if recorded is not None:
        # Checked here as well as inside `ensure_ready`, so a plan refuses a destination built
        # for another model rather than reporting a copy that the real run would then refuse.
        recorded.require_match(fingerprint)
    storage_name = target.storage_name(fingerprint)
    source_rows = await source.count()
    existing = await target.rows_in_space(fingerprint)
    if existing:
        _refuse_populated(storage_name, existing)
    if dry_run:
        return VectorMigration(
            storage_name=storage_name,
            generation=generation,
            dimension=fingerprint.dimension,
            source_rows=source_rows,
            copied=0,
            unverified=0,
            dry_run=True,
        )
    await target.ensure_ready(fingerprint)
    notify(
        f"copying {source_rows} vector(s) from {generation} into {storage_name} "
        f"at {fingerprint.dimension} dimensions"
    )
    copied, unverified = await _copy(source, target, report=notify, page_size=page_size)
    landed = await target.rows_in_space(fingerprint)
    if copied != source_rows or landed != source_rows:
        _refuse_short(storage_name, source_rows=source_rows, copied=copied, landed=landed)
    notify(f"copied {copied} vector(s) into {storage_name}")
    return VectorMigration(
        storage_name=storage_name,
        generation=generation,
        dimension=fingerprint.dimension,
        source_rows=source_rows,
        copied=copied,
        unverified=unverified,
        dry_run=False,
    )


async def _copy(
    source: InspectableVectorStore,
    target: AdoptingVectorStore,
    *,
    report: Callable[[str], None],
    page_size: int,
) -> tuple[int, int]:
    """Stream the source into the destination, verifying each row on the way through.

    **The verification is here rather than in a pass of its own**, and the cost of that is worth
    naming: a corrupt row is found after earlier rows have already been written, so the refusal
    leaves a partially populated collection behind. The alternative is a second full read of
    every vector before the first one moves, which doubles the work on every healthy corpus to
    improve the diagnostics on a damaged one — and the partial collection is cleared by the
    reset that the next attempt requires anyway, because a destination holding rows is refused.
    """
    copied = 0
    unverified = 0
    announced = 0
    async for page in source.inspection_pages(page_size=page_size):
        for row in page:
            verdict = row_integrity(row)
            if verdict is VectorIntegrity.UNVERIFIED:
                unverified += 1
            elif verdict is not VectorIntegrity.VERIFIED:
                _refuse_row(row, verdict)
        copied += await target.adopt_rows(page)
        if copied - announced >= PROGRESS_ROWS:
            announced = copied
            report(f"copied {copied} vector(s)")
    return copied, unverified


def _quiet(message: str) -> None:
    """The reporter for a migration nobody is watching."""
    del message


def _refuse_unbuilt() -> NoReturn:
    """Refuse a source that has never recorded which model its vectors came from."""
    msg = (
        "the source vector store records no embedding fingerprint, so there is nothing "
        "established about what its vectors mean and nothing here will guess. A store in that "
        "state has either never been written to or has lost the metadata beside its rows; "
        "`manicule doctor` reports which."
    )
    raise VectorMigrationError(msg)


def _refuse_short(storage_name: str, *, source_rows: int, copied: int, landed: int) -> NoReturn:
    """Refuse a copy that did not end with the destination holding what the source held.

    Counted from the destination rather than from the writes, because those answer different
    questions: ``copied`` is what this asked for and ``landed`` is what is there. A backend that
    silently coalesced two rows, a source that changed under the read, or a page that failed
    without raising all show up here and nowhere else — and a migration that reported success
    over a short destination would be discovered by a search that quietly returns less.
    """
    msg = (
        f"the copy did not reproduce the source: it holds {source_rows} row(s), this wrote "
        f"{copied}, and {storage_name} now holds {landed}. Something changed the source while "
        f"it was being read, or the destination did not keep everything it was given. Clear "
        f"the destination with `manicule reset-index` and run this again against a corpus "
        f"nothing else is writing to."
    )
    raise VectorMigrationError(msg)


def _refuse_populated(storage_name: str, existing: int) -> NoReturn:
    """Refuse a destination that already holds rows."""
    msg = (
        f"{storage_name} already holds {existing} vector(s), so nothing was copied. Merging "
        f"into it would mix this corpus with rows whose provenance nothing here can "
        f"establish — another installation sharing a namespace, or what an "
        f"interrupted migration left behind. Clear it deliberately with `manicule reset-index` "
        f"and run this again."
    )
    raise VectorMigrationError(msg)


def _refuse_row(row: Mapping[str, Any], verdict: VectorIntegrity) -> NoReturn:
    """Refuse a source row whose numbers no longer match what was recorded about them."""
    chunk = row.get(CHUNK_ID_COLUMN) or row.get(ID_COLUMN) or "an unnamed row"
    msg = (
        f"the vector stored for {chunk!r} reads as {verdict.value}, so the migration stopped "
        f"rather than carrying it. A row that fails its own checksum in the source fails it in "
        f"the destination too, and copying it would move the damage somewhere the original is "
        f"no longer there to be compared against. Run `manicule vector-checksum --verify` "
        f"against the embedded store to see the whole picture before moving the corpus."
    )
    raise VectorMigrationError(msg)


__all__ = [
    "MIGRATION_PAGE",
    "PROGRESS_ROWS",
    "VectorMigration",
    "migrate_vectors",
]
