"""The workspace boundary, in one place, for every relational surface that carries it.

:class:`SqliteDocStore` implements six protocols — documents and chunks, collections, tags,
versions, the trash, chunk relations — and every one of them is scoped to a single workspace
in exactly the same way: **the handle carries the workspace and no method takes one.** A query
cannot forget a parameter it was never given.

That shared state and the guards that use it live here rather than in the module that happens
to have needed them first. The alternative is five copies of a tenancy check, and a boundary
enforced in five places is a boundary enforced in whichever of them somebody remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import event, select

from manicule.core.errors import ManiculeError, UnknownEntityError
from manicule.storage import models
from manicule.storage.engine import session_factory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from manicule.core.retrieval import Filter

DEFAULT_WORKSPACE = "default"
"""Personal mode has one workspace and never says so out loud.

The workspace is bound to the store handle rather than passed per call, so a query cannot
forget it. Team mode supplies a real id; personal mode gets this one, and the isolation
predicate is identical either way — which means the path that enforces tenancy is exercised
by every test rather than only by the team-mode ones.
"""


class CrossWorkspaceCollisionError(ManiculeError):
    """A document id was offered to a workspace that does not own it.

    **This should be unreachable.** :func:`manicule.core.ids.document_id` takes the workspace
    as the first component of its digest, so two tenants indexing the same upstream source
    derive different ids by construction. What remains is a caller that built an id some other
    way, or a workspace mismatch between the handle and the document it was handed.

    Kept as a guard rather than deleted, because the failure it catches is silent: an id
    computed without the workspace lands on another tenant's row, overwriting content its
    author cannot read while its own document appears to vanish. An assertion that cannot fire
    costs one comparison per write; the same bug without it costs a tenant's data.
    """


class _CommitCounter:
    """Counts committed transactions on an engine.

    The invalidation signal behind the L1 query cache (``docs/retrieval.md`` §10.3), and it
    counts *commits* rather than calls to the write methods someone remembered to instrument.
    That is the same reasoning that puts FTS5 synchronization in triggers rather than in
    application code, and the same reasoning that puts the tenancy guards in this module: a
    per-method bump covers only the write paths its author enumerated, and the one nobody
    enumerated is the one that serves a stale ranking. A transaction that committed on this
    engine changed something; a read never reaches here, because a session closed without a
    commit rolls back.

    Over-counting is the safe direction and is accepted. A commit that could not have changed a
    result — a watermark, a connector's run metadata — still bumps the counter and costs at most
    a cold cache. Under-counting would serve a ranking computed over a corpus that no longer
    exists.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._value = 0
        event.listen(engine.sync_engine, "commit", self._committed)

    def _committed(self, connection: Connection) -> None:
        del connection  # the fact of the commit is the whole signal
        self._value += 1

    @property
    def value(self) -> int:
        return self._value


class WorkspaceScoped:
    """Shared construction and tenancy guards for the SQLite store's several surfaces.

    Not a protocol and not part of any public contract — it is the piece of implementation the
    protocol implementations have in common. Each mixin inherits it, so the concrete store has
    one ``__init__``, one session factory and one workspace however many protocols it satisfies.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        workspace_id: str = DEFAULT_WORKSPACE,
        sessions: async_sessionmaker[AsyncSession] | None = None,
        data_dir: Path | None = None,
        max_journal_records: int = 1_000_000,
        max_journal_metadata_bytes: int = 1024 * 1024 * 1024,
        max_acquired_blob_backlog_bytes: int = 20 * 1024 * 1024 * 1024,
        min_disk_headroom_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self._engine = engine
        self._workspace_id = workspace_id
        self._sessions = sessions or session_factory(engine)
        self._generation = _CommitCounter(engine)
        self._max_journal_records = max_journal_records
        self._max_journal_metadata_bytes = max_journal_metadata_bytes
        self._max_acquired_blob_backlog_bytes = max_acquired_blob_backlog_bytes
        database = engine.url.database
        if data_dir is not None:
            self._storage_root = data_dir
        elif database is None or database in {"", ":memory:"}:
            self._storage_root = None
        else:
            self._storage_root = Path(database).parent
        self._min_disk_headroom_bytes = min_disk_headroom_bytes

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def engine(self) -> AsyncEngine:
        """The engine this handle reads and writes through.

        Public because a process's composition root needs it and building a second one would
        be worse: two engines on the same file are two connection pools, two sets of pragmas
        and — for the migration that has to run before any query — two opinions about whether
        it already has. Migrations, backups and the sibling stores on this database all take
        the engine from whoever opened it.
        """
        return self._engine

    @property
    def generation(self) -> int:
        """Bumped by every committed transaction on this store's engine.

        Satisfies :class:`~manicule.core.retrieval.SupportsGeneration`, which is what lets the
        retrieval layer cache a ranking and know when to stop trusting it. It counts commits on
        the *engine*, so a write through any other handle on the same database — another
        workspace's store, an ingest run, a repair verb, a collection being renamed — moves this
        one too. That is deliberate: this counter's job is to be impossible to bypass, not to be
        minimal.

        Here rather than on the document store for the same reason everything else in this class
        is: six protocols share one handle, and several of them write. A counter attached to one
        surface would miss the commits of the other five.
        """
        return self._generation.value

    async def ensure_workspace(self) -> None:
        """Create this store's workspace row if it is absent. Idempotent."""
        async with self._sessions.begin() as session:
            existing = await session.get(models.Workspace, self._workspace_id)
            if existing is None:
                session.add(
                    models.Workspace(id=self._workspace_id, name=self._workspace_id, settings={})
                )

    # --- tenancy guards ---------------------------------------------------------------------

    def _require_honorable(
        self,
        filter: Filter | None,  # noqa: A002
        honored: frozenset[str],
        operation: str,
    ) -> None:
        """Refuse a filter this query cannot apply in full.

        Two refusals, and the order matters only in that both must happen before the
        statement is built.

        **A workspace this handle does not serve.** A subset check, because
        :attr:`~manicule.core.retrieval.Filter.workspace_ids` is set-valued: cross-workspace
        search is N scoped handles fanned out and merged (``docs/retrieval.md`` §3.2), so this
        handle only ever sees its own workspace. A filter naming another one is a caller who
        believes they are querying somewhere else, and answering the question they did not ask
        is the shape a cross-tenant leak takes.

        **A field this query has no column for.** Silently dropping it returns rows the filter
        was written to exclude — a listing or a search that still looks like it worked. The
        same rule as :func:`~manicule.storage.vectors.predicate_for`, applied here so that
        "a store honors the whole filter or refuses" holds for both halves of storage.

        Raises:
            CrossWorkspaceCollisionError: The filter names another workspace.
            ValueError: The filter sets a field this query cannot apply.
        """
        if filter is None:
            return

        outside = sorted(filter.workspace_ids - {self._workspace_id})
        if outside:
            named = ", ".join(repr(name) for name in outside)
            msg = (
                f"filter names workspace(s) {named} but this store serves "
                f"{self._workspace_id!r}. Cross-workspace search is one store per workspace, "
                f"merged — open a store for each."
            )
            raise CrossWorkspaceCollisionError(msg)

        unhonored = sorted(filter.restricting_fields - honored)
        if unhonored:
            msg = (
                f"this store cannot {operation} by {', '.join(unhonored)}. Resolve those "
                f"fields into document_ids first; applying the rest and ignoring them would "
                f"return rows the filter was written to exclude."
            )
            raise ValueError(msg)

    # --- row lookups ------------------------------------------------------------------------

    async def _live_document(
        self, session: AsyncSession, document_id: str
    ) -> models.Document | None:
        row = await session.get(models.Document, document_id)
        if row is None or row.workspace_id != self._workspace_id or row.deleted_at is not None:
            return None
        return row

    async def _any_document(
        self, session: AsyncSession, document_id: str
    ) -> models.Document | None:
        """A document in this workspace whether or not it is in the trash.

        The trash surface and the version history both have to see deleted rows — that is what
        they are for — so they cannot go through :meth:`_live_document`, which exists to keep
        a soft-deleted document out of everything else.
        """
        row = await session.get(models.Document, document_id)
        if row is None or row.workspace_id != self._workspace_id:
            return None
        return row

    async def _require_live_documents(
        self, session: AsyncSession, document_ids: Sequence[str]
    ) -> list[str]:
        """The ids, having checked every one of them is a live document in this workspace.

        **All or nothing.** A membership write that skipped the ids it could not see would
        report "added forty" having added thirty-nine, and the one it dropped is the one that
        mattered — a typo, or a document id from another tenant. Refusing the batch is the only
        outcome a caller can act on.

        Raises:
            UnknownEntityError: Naming the ids this handle cannot see, without saying which of
                them exist elsewhere.
        """
        wanted = list(dict.fromkeys(document_ids))
        if not wanted:
            return []
        found = set(
            (
                await session.execute(
                    select(models.Document.id).where(
                        models.Document.id.in_(wanted),
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        missing = [document_id for document_id in wanted if document_id not in found]
        if missing:
            named = ", ".join(repr(document_id) for document_id in missing)
            msg = (
                f"no live document {named} in workspace {self._workspace_id!r}. Nothing was "
                f"written: a partially applied membership change reports a success it did not "
                f"have."
            )
            raise UnknownEntityError(msg)
        return wanted


__all__ = [
    "DEFAULT_WORKSPACE",
    "CrossWorkspaceCollisionError",
    "WorkspaceScoped",
]
