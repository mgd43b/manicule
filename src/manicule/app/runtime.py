"""The composition root for a process: configuration and plugins into a running system.

:class:`Runtime` is the production :class:`~manicule.app.ports.Backend`. It owns the container,
the database engine and the lifecycle, and it builds each expensive part **the first time it is
asked for and never before** — so ``manicule doctor`` loads no model runtime, ``manicule
document list`` loads no provider library, and a completion script loads neither.

It contains no policy. Everything a surface can decide is in
:class:`~manicule.app.service.ApplicationService`; everything a plugin can decide is in the
container. What is left here is wiring, and wiring that is only ever exercised one way is
wiring nobody can test — which is why the service takes a protocol and this is one
implementation of it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import shutil
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast, override

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from manicule.app.results import ApiKeySummary, Check, CheckState
from manicule.config.loader import load_settings
from manicule.container import keys
from manicule.container.container import Container, build_container
from manicule.core.content import RawDocument
from manicule.core.errors import ManiculeError, PolicyError, UnknownEntityError
from manicule.core.lifecycle import HealthState
from manicule.ingest.capacity import CapacityRefusedError
from manicule.ingest.recovery import InstanceLock
from manicule.plugins.manifest import ComponentKind

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.app.ports import (
        Answering,
        Conversing,
        DocumentSurface,
        Ingesting,
        Keys,
        Maintenance,
        Organizing,
        ResetOutcome,
        Retrieving,
        Telemetry,
    )
    from manicule.config.settings import Settings
    from manicule.core.acquisition import AcquisitionRun
    from manicule.core.fingerprints import GlossaryFingerprint
    from manicule.core.protocols import Connector, Embedder, Parser, VectorStore
    from manicule.core.source_lifecycle import LifecycleOutcome, LifecyclePlan
    from manicule.ingest.middleware import MiddlewareRunner
    from manicule.ingest.pipeline import BlobSink, IngestPipeline, RunReport, Watching
    from manicule.ingest.ports import IngestStore
    from manicule.ingest.reembed import ReembedPlan, ReembedRecovery, ReembedRun
    from manicule.ingest.reindex import GlossarySweep, ReindexReport, StaleSweep
    from manicule.plugins.registry import Discovery
    from manicule.storage.docstore import SqliteDocStore
    from manicule.storage.reembed import (
        LanceShadowGenerations,
        SqliteReembedCorpus,
        SqliteReembedStore,
    )

ARCHIVE_MANIFEST = "manicule-export.json"
"""The file that makes an exported directory an archive rather than a pile of blobs."""

ARCHIVE_VERSION = 1
"""Bumped when the archive layout changes in a way an older import cannot read."""

_DERIVED_MUTATION_GUARDS: ContextVar[frozenset[int]] = ContextVar(
    "manicule_derived_mutation_guards", default=frozenset()
)


async def _recover_reembed_runs(
    run_ids: Sequence[str], resume: Callable[[str], Awaitable[object]]
) -> ReembedRecovery:
    """Attempt every run independently and expose only aggregate error classes."""
    from manicule.ingest.reembed import ReembedRecovery  # noqa: PLC0415

    recovered = 0
    failures = 0
    failure_types: set[str] = set()
    for run_id in run_ids:
        try:
            await resume(run_id)
        except Exception as error:  # noqa: BLE001 - each durable run is an isolation boundary
            failures += 1
            failure_types.add(type(error).__name__)
        else:
            recovered += 1
    return ReembedRecovery(
        recovered=recovered,
        failures=failures,
        failure_types=tuple(sorted(failure_types)),
    )


KEY_PREFIX = "mnk_"
"""What every API key starts with.

A recognizable prefix so a leaked key is greppable — by its owner, and by a secret scanner
that has been taught the pattern. The prefix plus six characters is what is stored in the
clear, which is enough to tell two keys apart and not enough to be one.
"""


class AssemblyError(ManiculeError):
    """Something the runtime could not assemble."""


class ArchiveEntry(BaseModel):
    """One document in an exported archive: where it came from, and where its bytes are."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = "import"
    source_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    title: str = ""
    media_type: str = Field(min_length=1)
    version_token: str | None = None
    blob: str = Field(min_length=1, description="Path to the retained bytes, inside the archive.")


class ArchiveManifest(BaseModel):
    """What an export wrote, and what an import validates before reading a single byte.

    **No chunks and no vectors, by construction.** There is nowhere in this model to put
    them, so an archive cannot carry an index built by another chunker or another embedder
    into a store whose fingerprints say something else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    workspace: str = ""
    documents: tuple[ArchiveEntry, ...] = ()


@dataclass(slots=True)
class _Lazy:
    """One built-once component, with the lock that keeps it built once.

    A plain ``if self._x is None`` is a race the moment two tool calls arrive together, and
    the thing being built here is a model runtime — building it twice is not a wasted object,
    it is a second copy of a multi-gigabyte model.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    value: object | None = None


class Runtime:
    """A whole manicule, assembled from configuration and what plugins registered."""

    def __init__(
        self, settings: Settings, *, discovery: Discovery | None = None, writer: bool = True
    ) -> None:
        self._settings = settings
        self._container = build_container(settings, discovery=discovery)
        self._slots: dict[str, _Lazy] = {}
        self._engine: AsyncEngine | None = None
        self._migrated = False
        self._writer = writer
        self._lock: InstanceLock | None = None
        self._derived_mutation_lock = asyncio.Lock()

    # --- lifecycle --------------------------------------------------------------------------

    @classmethod
    def open(cls, *, writer: bool = True, **overrides: Any) -> Runtime:  # noqa: ANN401 - mirrors Settings
        """Load configuration, verify it against what is installed, and return the runtime.

        Args:
            writer: Whether this process may mutate the data directory. **The default, and it
                is the default deliberately**: a caller that has not thought about it gets the
                exclusion rather than silently going without, and the cost of being wrong that
                way round is a refusal an operator can read. ``docs/ingest.md`` 8.6 has the
                classification and :func:`~manicule.app.dispatch.writes` applies it.
            **overrides: Settings fields.

        Raises:
            ConfigError: The configuration is malformed.
            PolicyError: It is individually valid and jointly unrunnable, or it names a
                component nothing installed provides. Everything wrong is listed at once.
        """
        return cls(load_settings(**overrides), writer=writer)

    async def __aenter__(self) -> Self:
        """Take the data directory, if this runtime is one that writes.

        **Here rather than at the first write, and that ordering is the whole point.** The lock
        has to be held before recovery, before the schema migration and before anything opens
        the database — a lock acquired after recovery has already requeued another process's
        in-flight documents has protected nothing that mattered. This is the earliest moment
        that exists: the runtime is constructed and nothing has been resolved yet.

        There is no check-then-acquire gap because there is no check. The lock is taken
        unconditionally for a writer, and `flock` either grants it or does not.

        Raises:
            InstanceLockedError: Another process holds this data directory.
        """
        self.acquire()
        return self

    def acquire(self) -> None:
        """Take the data directory now rather than when the runtime is entered. Idempotent.

        For the callers that are servers. A long-lived process wants the refusal *before* it
        starts announcing itself — before a port is bound, before a banner is printed — and it
        wants it as its own one-line report rather than as a traceback out of an ``async with``
        whose body is a hundred lines of serving. Calling this first gets that; entering the
        runtime afterwards finds the lock already held and does nothing.

        A reader takes nothing here and nothing later, which is what makes this safe to call
        unconditionally: whether it locks is the runtime's decision, not the caller's.

        Raises:
            InstanceLockedError: Another process holds this data directory.
        """
        if self._writer and self._lock is None:
            self._lock = InstanceLock(self._settings.data_dir).acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, tb
        await self.aclose(pending=exc)

    async def aclose(self, *, pending: BaseException | None = None) -> None:
        """Tear everything down, and dispose the engine last.

        The engine is disposed after the container because a component's teardown may still
        write — a cache flush, a final status — and an engine disposed first turns that into a
        connection error during shutdown, which is reported as a teardown failure and is not
        one.
        """
        try:
            close_error: Exception | None = None
            try:
                await self._container.aclose()
            except Exception as error:  # noqa: BLE001 - teardown must still close vector handles
                close_error = error
            try:
                vectors = self._slots.get("vectors")
                if vectors is not None and vectors.value is not None:
                    from manicule.storage.vectors import (  # noqa: PLC0415
                        PublishedLanceVectorStore,
                    )

                    if isinstance(vectors.value, PublishedLanceVectorStore):
                        await vectors.value.teardown()
            except Exception as vector_error:
                if close_error is None:
                    raise
                close_error.add_note(
                    f"closing the published vector handle also failed: {vector_error}"
                )
            if close_error is not None:
                raise close_error  # noqa: TRY301 - preserve the container's original exception
        except Exception as during_teardown:
            if pending is None:
                raise
            pending.add_note(f"shutting down also failed: {during_teardown}")
        finally:
            try:
                if self._engine is not None:
                    await self._engine.dispose()
                    self._engine = None
            finally:
                # Released last, after the container and after the engine, so the lock outlives
                # every storage operation it was taken to exclude. Giving it up first would
                # open the directory to a second writer while this one was still flushing —
                # which is the window the lock exists to close, moved to the end of the run.
                if self._lock is not None:
                    self._lock.release()
                    self._lock = None

    # --- what the service is given ----------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def container(self) -> Container:
        """Configured component graph shared by ingestion and offline derivation assembly."""
        return self._container

    @property
    def workspace(self) -> str:
        return self._settings.workspace

    @property
    def derived_mutation_lock(self) -> asyncio.Lock:
        """Process-local half of the filesystem-pinned derived mutation barrier."""
        return self._derived_mutation_lock

    @asynccontextmanager
    async def derived_mutation_guard(self) -> AsyncGenerator[None]:
        """Serialize a whole derived publication lifecycle against reset, reentrantly."""
        key = id(self._derived_mutation_lock)
        held = _DERIVED_MUTATION_GUARDS.get()
        if key in held:
            yield
            return
        async with self._derived_mutation_lock:
            token = _DERIVED_MUTATION_GUARDS.set(held | {key})
            try:
                yield
            finally:
                _DERIVED_MUTATION_GUARDS.reset(token)

    @property
    def discovery(self) -> Discovery | None:
        return self._container.discovery

    @property
    def container_view(self) -> object:
        """The container, for a caller that legitimately needs to resolve something else."""
        return self._container

    async def documents(self) -> DocumentSurface:
        """The relational store, migrated before it is ever read."""
        return await self._once("documents", self._build_documents)

    async def vectors(self) -> VectorStore:
        """The vector store, opened but not yet committed to a dimension.

        Deliberately unprepared. Deleting from a store that has never held a vector is a
        no-op, and preparing one would mean loading the embedding model to find out its
        dimension — which is a multi-second, multi-gigabyte cost for a command that is only
        removing rows.
        """
        return await self._once("vectors", self._build_vectors)

    async def vector_directory(self) -> Path:
        """Physical vector root selected by this workspace's compatibility marker."""
        from sqlalchemy import select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import VECTORS_DIRNAME  # noqa: PLC0415
        from manicule.storage.vectors import workspace_vector_directory  # noqa: PLC0415

        await self.documents()
        root = self._settings.data_dir / VECTORS_DIRNAME
        async with self.require_engine().connect() as connection:
            namespace = (
                await connection.execute(
                    select(models.IndexState.vector_namespace).where(
                        models.IndexState.workspace_id == self.workspace
                    )
                )
            ).scalar_one_or_none()
        return root if namespace == "legacy" else workspace_vector_directory(root, self.workspace)

    async def derived_reset_epoch(self) -> int:
        """Return the durable workspace fence captured by newly assembled derived writers."""
        from sqlalchemy import select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415

        await self.documents()
        async with self.require_engine().connect() as connection:
            epoch = (
                await connection.execute(
                    select(models.Workspace.derived_reset_epoch).where(
                        models.Workspace.id == self.workspace
                    )
                )
            ).scalar_one()
        return int(epoch)

    async def invalidate_derived_runtime(self) -> None:
        """Close and evict every cached object that can retain an old index identity."""
        pipeline = self._slots.get("pipeline")
        if pipeline is not None and pipeline.value is not None:
            await cast("IngestPipeline", pipeline.value).aclose()
        vectors = self._slots.get("vectors")
        if vectors is not None and vectors.value is not None:
            from manicule.storage.vectors import PublishedLanceVectorStore  # noqa: PLC0415

            if isinstance(vectors.value, PublishedLanceVectorStore):
                await vectors.value.teardown()
        for slot in ("pipeline", "prepared_vectors", "vectors", "retriever", "answerer"):
            self._slots.pop(slot, None)

    async def embedder(self) -> Embedder:
        """The configured embedder, exposed for index-maintenance orchestration."""
        return await self._container.aget(keys.EMBEDDER)

    async def prepared_vectors(self) -> VectorStore:
        """The vector store, committed to the configured embedder's space.

        Both paths that touch vectors for real — ingest and retrieval — come through here, and
        this is the only place ``ensure_ready`` is called. Without it the store never learns
        which space it holds: the first upsert fails, the document is recorded ``failed`` at
        the ``store`` stage, and nothing about the message says the index was never prepared.

        It is also where a directory holding another model's vectors is refused, before a
        single vector is written into a space it does not belong to.
        """
        async with self.derived_mutation_guard():
            return await self._once("prepared_vectors", self._build_prepared_vectors)

    async def retriever(self) -> Retrieving:
        """The whole of retrieval. Builds the embedder, so it is not built for a listing."""
        return await self._once("retriever", self._build_retriever)

    async def answerer(self) -> Answering:
        """The answer path. Builds the generator, so it is not built for a search."""
        return await self._once("answerer", self._build_answerer)

    async def ingestion(self) -> Ingesting:
        """The ingest operations. The pipeline itself is built on first use."""
        return await self._once("ingestion", self._build_ingestion)

    async def maintenance(self) -> Maintenance:
        """Backup, restore, export and reset, over this runtime's engine."""
        return await self._once("maintenance", self._build_maintenance)

    async def organization(self) -> Organizing:
        """Collections, tags and the trash.

        The **same object** the document store is. Two handles over one workspace would be two
        opinions about which documents are live, and the collection reads join against exactly
        the predicate the document reads use.
        """
        return await self._once("organization", self._build_organization)

    async def conversations(self) -> Conversing:
        """Conversations, turns and share links, over this runtime's engine.

        One slot, shared with the answer path. A second store over the same engine would be a
        second session factory writing turns into the same rows, and — more to the point — a
        second place a share link could be minted from.
        """
        return await self._once("conversations", self._build_conversations)

    async def telemetry(self) -> Telemetry:
        """Query logs and the audit trail."""
        return await self._once("telemetry", self._build_telemetry)

    async def keys(self) -> Keys:
        """API keys for this workspace."""
        return await self._once("keys", self._build_keys)

    async def component_checks(self) -> Sequence[Check]:
        """Health of what is already constructed. Constructs nothing.

        ``doctor`` runs before anything is built in the common case, so this is usually empty
        — and an empty list is the honest answer to "how are the components you have not made
        yet", where a fabricated "ok" would be a diagnostic that reports health it never
        measured.
        """
        report = await self._container.health()
        return [
            Check(
                name=f"component:{check.name}",
                state=_state_name(check.state),
                detail=check.detail,
            )
            for check in sorted(report.checks, key=lambda check: check.name)
        ]

    # --- construction -----------------------------------------------------------------------

    async def _once[T](self, slot: str, build: Callable[[], Awaitable[T]]) -> T:
        """Build ``slot`` once, under a lock, and hand back the same object thereafter."""
        lazy = self._slots.setdefault(slot, _Lazy())
        if lazy.value is not None:
            return cast("T", lazy.value)
        async with lazy.lock:
            if lazy.value is None:
                lazy.value = await build()
            return cast("T", lazy.value)  # pyright: ignore[reportUnnecessaryCast] - the slot is `object`

    async def _build_documents(self) -> DocumentSurface:
        # Imported here, not at module scope: Alembic is not a cheap import and a process
        # that never opens the index should not pay for it.
        from manicule.app.ports import DocumentSurface as Surface  # noqa: PLC0415

        store = await self._container.aget(keys.DOC_STORE)
        # Checked rather than cast. The container types this by `DocStore`, and the surface
        # needs more of it than that protocol promises; a cast would turn a store missing a
        # method into an AttributeError from inside a tool call.
        if not isinstance(store, Surface):
            msg = (
                f"the configured document store {type(store).__name__} does not provide the "
                f"reads the surfaces need (listing, counting, statistics, soft delete)."
            )
            raise AssemblyError(msg)
        engine = getattr(store, "engine", None)
        if engine is None:  # pragma: no cover - a store that is not the built-in one
            msg = (
                "the configured document store does not expose its engine, so migrations and "
                "backups have nothing to run against"
            )
            raise AssemblyError(msg)
        self._engine = engine
        await self._initialize_storage(store, engine)
        return store

    async def _initialize_storage(self, store: object, engine: AsyncEngine) -> None:
        """Migrate the schema and make sure this workspace has a row. Both are writes.

        **This is the boundary a reader can cross**, and naming it is most of what makes the
        reader classification honest. A command that only reads still has to meet a database
        that is at the right revision and a workspace row that exists, and on a directory that
        has never been used neither is true — so the first ``manicule search`` after an upgrade
        performs a migration, which is emphatically a write.

        So a reader does it **under the writer's lock**, and only when there is something to
        do. On an established directory both steps are no-ops, the check costs two statements,
        and no lock is taken at all — which is what keeps `doctor` and `search` runnable while
        a sweep holds the directory. On a directory that needs initializing, a reader takes the
        lock for exactly the length of the initialization and gives it back; if a writer holds
        it, the reader is refused with the same message a second writer gets, which is the
        honest answer because it genuinely cannot proceed.

        A writer is already holding the lock from :meth:`__aenter__` and simply proceeds.
        """
        from manicule.storage.migrator import upgrade  # noqa: PLC0415 - Alembic is not cheap

        ensure = getattr(store, "ensure_workspace", None)
        if self._writer:
            if not self._migrated:
                # Before the first read, always. A query against an un-migrated database fails
                # in whatever way SQLite happens to fail, at whichever statement happens to run
                # first.
                await upgrade(engine)
                self._migrated = True
            if ensure is not None:
                await ensure()
            # This is a storage-data migration rather than an Alembic schema rewrite: it must
            # validate content-addressed files before it can make them manifest GC roots.  A
            # writer holds the instance lock here, and no connector is constructed or called.
            from manicule.storage.blobs import BlobStore  # noqa: PLC0415
            from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
            from manicule.storage.legacy_snapshots import (  # noqa: PLC0415
                migrate_legacy_snapshots,
            )

            if isinstance(store, SqliteDocStore):
                migrated = await migrate_legacy_snapshots(
                    store,
                    BlobStore(engine, self._settings.data_dir),
                )
                if migrated.deferred:
                    msg = (
                        "legacy retained-source ownership is still leased by another process; "
                        "writer startup refused"
                    )
                    raise RuntimeError(msg)
            return
        if not self._migrated:
            if await self._storage_needs_initializing(engine):
                with InstanceLock(self._settings.data_dir):
                    await upgrade(engine)
            self._migrated = True
        if ensure is not None:
            # Outside the lock, and the docstring above says why: one guarded insert of a row
            # nobody else is inserting, which no concurrent writer can be harmed by.
            await ensure()

    async def _storage_needs_initializing(self, engine: AsyncEngine) -> bool:
        """Whether the schema is behind head. A read, and the only thing that decides the lock.

        Deliberately narrower than "would the initialization write anything". Creating a
        workspace row is a write too, but it is one statement guarded by its own transaction
        and it cannot corrupt anything a concurrent writer is doing; a migration rebuilds
        tables. Taking an exclusive lock to insert a row nobody else is inserting would refuse
        a reader for no gain, which is the failure mode this whole classification exists to
        avoid.
        """
        from manicule.storage.migrator import current_revision, head_revision  # noqa: PLC0415

        async with engine.connect() as connection:
            at = await connection.run_sync(current_revision)
        return at != head_revision()

    async def _build_vectors(self) -> VectorStore:
        from manicule.storage.vectors import (  # noqa: PLC0415
            LanceVectorStore,
            PublishedLanceVectorStore,
        )

        await self.documents()
        store = await self._container.aget(keys.VECTOR_STORE)
        if isinstance(store, LanceVectorStore):
            from sqlalchemy import select  # noqa: PLC0415

            from manicule.storage import models  # noqa: PLC0415

            async with self.require_engine().connect() as connection:
                row = (
                    await connection.execute(
                        select(
                            models.IndexState.vector_namespace,
                            models.Workspace.derived_reset_epoch,
                        )
                        .select_from(models.Workspace)
                        .outerjoin(
                            models.IndexState,
                            models.IndexState.workspace_id == models.Workspace.id,
                        )
                        .where(models.Workspace.id == self.workspace)
                    )
                ).one()
            identity_namespace = row.vector_namespace
            return PublishedLanceVectorStore(
                await self.vector_directory(),
                self.require_engine(),
                workspace_id=self.workspace,
                identity_namespace=(
                    None if identity_namespace is None else str(identity_namespace)
                ),
                expected_reset_epoch=int(row.derived_reset_epoch),
            )
        return store

    async def _build_prepared_vectors(self) -> VectorStore:
        store = await self.vectors()
        embedder = await self._container.aget(keys.EMBEDDER)
        await store.ensure_ready(embedder.fingerprint)
        return store

    async def _build_retriever(self) -> Retrieving:
        from manicule.retrieval.retriever import build_retriever  # noqa: PLC0415 - heavy

        await self.documents()
        # Before the dense leg is ever asked a question. A store that has not been prepared
        # raises rather than returning nothing, so an empty index would answer every search
        # with an error about vector spaces.
        vectors = await self.prepared_vectors()
        return await build_retriever(self._container, vectors=vectors)

    async def _build_answerer(self) -> Answering:
        from manicule.generation.answering import Answerer  # noqa: PLC0415 - heavy
        from manicule.generation.policy import EgressPolicy  # noqa: PLC0415
        from manicule.generation.redaction import Redactor  # noqa: PLC0415
        from manicule.generation.verification import (  # noqa: PLC0415
            ChainRouter,
            CitationVerifier,
            RetainedBytesResolver,
        )
        from manicule.storage.conversations import SqliteConversationStore  # noqa: PLC0415

        settings = self._settings
        store = await self.documents()
        generator = await self._container.aget(keys.GENERATOR)
        resolver = RetainedBytesResolver(
            blobs=await self.blobs(),
            router=ChainRouter(chain=_ContainerChain(self._container)),
        )
        # The same handle the surfaces use, not a second one. Two stores over one engine is
        # two places a share link can be minted from and two opinions about what is deleted.
        conversations = cast("SqliteConversationStore", await self.conversations())
        return Answerer(
            generator=generator,
            verifier=CitationVerifier(resolver, timeout_s=settings.llm.citation_verify_timeout_s),
            documents=store,
            settings=settings,
            policy=EgressPolicy.of(settings, settings.workspace),
            redactor=Redactor(settings.security.data_policy.auto_redact),
            conversations=conversations,
        )

    async def _build_ingestion(self) -> Ingesting:
        return _Ingestion(self)

    async def _build_maintenance(self) -> Maintenance:
        await self.documents()
        return _Maintenance(self)

    async def _build_organization(self) -> Organizing:
        from manicule.app.ports import Organizing as Surface  # noqa: PLC0415

        store = await self.documents()
        # Checked rather than cast, like the document surface itself. A store that does not
        # provide collections would otherwise become an AttributeError from inside a route.
        if not isinstance(store, Surface):
            msg = (
                f"the configured document store {type(store).__name__} does not provide "
                f"collections, tags and the trash, which the surfaces need."
            )
            raise AssemblyError(msg)
        return store

    async def _build_conversations(self) -> Conversing:
        from manicule.storage.conversations import SqliteConversationStore  # noqa: PLC0415

        await self.documents()
        store = SqliteConversationStore(
            self.require_engine(), workspace_id=self._settings.workspace
        )
        # Every table here cascades from `workspaces`, so a handle bound to a workspace with
        # no row is a foreign-key failure on the first conversation rather than at startup.
        await store.ensure_workspace()
        return store

    async def _build_telemetry(self) -> Telemetry:
        await self.documents()
        return _Telemetry(self)

    async def _build_keys(self) -> Keys:
        await self.documents()
        return _Keys(self)

    async def blobs(self) -> BlobSink:
        """Where retained source bytes live, or the sink that keeps none.

        Public because two collaborators need the *same* one: re-parse reads bytes back
        through it and export copies them out of it, and a second blob store over the same
        directory would be a second opinion about what has been retained.
        """
        return await self._once("blobs", self._build_blobs)

    async def _build_blobs(self) -> BlobSink:
        from manicule.ingest.pipeline import NoRetention  # noqa: PLC0415
        from manicule.storage.blobs import BlobStore  # noqa: PLC0415

        await self.documents()
        if not self._settings.storage.retain_source_bytes:
            return NoRetention()
        return BlobStore(
            self.require_engine(),
            self._settings.data_dir,
            min_disk_headroom_bytes=self._settings.ingest.min_disk_headroom_bytes,
            max_acquired_blob_backlog_bytes=(self._settings.ingest.max_acquired_blob_backlog_bytes),
        )

    async def pipeline(self) -> IngestPipeline:
        """The ingest pipeline, refused before construction if it cannot write to this index."""
        async with self.derived_mutation_guard():
            return await self._once("pipeline", self._build_pipeline)

    async def acquisition_pipeline(self) -> IngestPipeline:
        """The source-only pipeline, assembled without requesting any derived component."""
        return await self._once("acquisition_pipeline", self._build_acquisition_pipeline)

    async def _build_acquisition_pipeline(self) -> IngestPipeline:
        from manicule.ingest.pipeline import IngestPipeline  # noqa: PLC0415
        from manicule.ingest.ports import AcquisitionStore as AcquisitionSurface  # noqa: PLC0415
        from manicule.ingest.ports import FencedIngestStore  # noqa: PLC0415

        settings = self._settings
        if not settings.storage.retain_source_bytes:
            raise AssemblyError("acquire-only requires retained source bytes")
        store = await self.documents()
        if not isinstance(store, AcquisitionSurface) or not isinstance(store, FencedIngestStore):
            msg = (
                f"the configured document store {type(store).__name__} does not provide durable "
                "source acquisition"
            )
            raise AssemblyError(msg)
        return IngestPipeline(
            store=store,  # pyright: ignore[reportArgumentType] - checked surfaces above
            acquisitions=store,
            workspace=settings.workspace,
            blobs=await self.blobs(),
            fetch_concurrency=settings.ingest.fetch_concurrency,
            parse_workers=1,
            queue_depth_factor=settings.ingest.queue_depth_factor,
            shutdown_grace_s=settings.ingest.shutdown_grace_s,
            max_fetch_bytes=settings.ingest.max_fetch_bytes,
            snapshot_policy=settings.ingest.snapshot_promotion_policy,
            mutation_guard=self.derived_mutation_guard,
        )

    async def _build_pipeline(self) -> IngestPipeline:
        from manicule.ingest.pipeline import IngestPipeline  # noqa: PLC0415
        from manicule.ingest.ports import (  # noqa: PLC0415
            AcquisitionStore as AcquisitionSurface,
        )
        from manicule.ingest.ports import FencedIngestStore  # noqa: PLC0415
        from manicule.ingest.refusals import check_before_run  # noqa: PLC0415
        from manicule.ingest.workers import WorkerPool, worker_config  # noqa: PLC0415

        settings = self._settings
        store = await self.documents()
        expected_reset_epoch = (
            await self.derived_reset_epoch()
            if getattr(store, "assert_derived_reset_epoch", None) is not None
            else None
        )
        acquisitions: AcquisitionSurface | None = None
        if settings.storage.retain_source_bytes:
            # Retention enables the durable journal path. Third-party document stores may
            # satisfy the application surfaces without implementing that separate protocol;
            # refuse them at assembly rather than failing inside a live ingest with an
            # AttributeError after source work has begun.
            if not isinstance(store, AcquisitionSurface) or not isinstance(
                store, FencedIngestStore
            ):
                msg = (
                    f"the configured document store {type(store).__name__} does not provide "
                    "durable source acquisition; disable source-byte retention or configure "
                    "a store that implements AcquisitionStore and FencedIngestStore"
                )
                raise AssemblyError(msg)
            acquisitions = store
        chunker = await self._container.aget(keys.CHUNKER)
        embedder = await self._container.aget(keys.EMBEDDER)
        vectors = await self.prepared_vectors()
        # Through the accessor rather than assembled here, so that the pipeline's chain and the
        # one glossary repair compares against are one reading of one configuration. Two
        # readings that agreed today would part company the first time either grew a filter.
        middleware = await self.middleware()
        fingerprint = chunker.fingerprint.with_middleware(middleware.declarations())
        # Prepared a second time, now that the middleware declarations are known. `ensure_ready`
        # is idempotent and the fingerprint is the same one; what this adds is the declaration
        # set the store folds into every embedding-input identity it writes. It is done here
        # rather than in `_build_prepared_vectors` because ingest is the only path that writes
        # vectors, and making retrieval load the middleware registry to answer a query would be
        # a cost paid by the wrong caller.
        await vectors.ensure_ready(
            embedder.fingerprint, embed_text_middleware=fingerprint.embed_text_middleware
        )
        # Before a single document is fetched. An index built by a different chunker or a
        # different embedder is refused here rather than discovered halfway through a run,
        # by which point half the corpus disagrees with the other half.
        await check_before_run(
            embed=embedder.fingerprint,
            chunk=fingerprint,
            # The concrete store satisfies both surfaces; the container hands it back typed by
            # the narrower one, and nothing at this seam can check the wider one statically.
            # `_build_documents` asserts the reads the surfaces need at construction time.
            store=cast("IngestStore", store),
            vectors=vectors,
            expected_reset_epoch=expected_reset_epoch,
        )
        pool = WorkerPool(
            worker_config(settings, chunker=chunker, embedder=embedder),
            workers=settings.ingest.parse_workers,
            timeout_s=settings.ingest.parse_timeout_s,
            poll_interval_s=settings.ingest.memory_poll_interval_s,
            max_documents=settings.ingest.max_documents_per_worker,
        )
        return IngestPipeline(
            store=store,  # pyright: ignore[reportArgumentType] - the store satisfies IngestStore
            acquisitions=acquisitions,
            chunker=chunker,
            embedder=embedder,
            vectors=vectors,
            runner=pool,
            resolve_chain=self._container.parser_chain_names,
            middleware=middleware,
            chunk_fingerprint=fingerprint,
            workspace=settings.workspace,
            blobs=await self.blobs(),
            fetch_concurrency=settings.ingest.fetch_concurrency,
            # The three that size the staged run, and all three were configurable and inert
            # until they arrived here. `parse_workers` is passed rather than read off the pool
            # because the pipeline sizes its own hand-offs from it and a `ParseRunner` is a
            # protocol with no size on it — reading one structurally would be a guess about the
            # implementation rather than a reading of the configuration.
            parse_workers=pool.size,
            queue_depth_factor=settings.ingest.queue_depth_factor,
            shutdown_grace_s=settings.ingest.shutdown_grace_s,
            max_fetch_bytes=settings.ingest.max_fetch_bytes,
            target_batch_tokens=settings.ingest.target_batch_tokens,
            max_embed_batch=settings.ingest.max_embed_batch,
            snapshot_policy=settings.ingest.snapshot_promotion_policy,
            mutation_guard=self.derived_mutation_guard,
            expected_reset_epoch=expected_reset_epoch,
            # Passed rather than defaulted, which it had been since the setting shipped. The
            # field is documented as the switch an operator throws while investigating a
            # detector that is producing rubbish, and nothing outside `settings.py` read it: a
            # configuration saying `detect_on_ingest = false` detected on ingest anyway. It has
            # to reach here now, because the glossary fingerprint records enablement, and a
            # fingerprint that recorded a state configuration could not actually reach would be
            # a lie told in a column rather than a setting quietly doing nothing.
            detect_glossary=settings.rag.glossary.detect_on_ingest,
        )

    async def middleware(self) -> MiddlewareRunner:
        """The configured hook chain, as the pipeline would run it.

        Public because glossary repair needs the same chain the pipeline folds in, and reading
        it a second way is how two paths end up computing two fingerprints for one
        configuration — which would make every document stale the moment either of them ran.
        """
        from manicule.ingest.middleware import MiddlewareRunner  # noqa: PLC0415

        return MiddlewareRunner(await self._container.middleware())

    def require_engine(self) -> AsyncEngine:
        """The engine the document store opened.

        Public for the same reason :meth:`blobs` is: migrations, backups and the conversation
        store all have to be on the *same* engine, and a second one over the same file is a
        second connection pool with its own opinion about whether the schema is current.

        Raises:
            AssemblyError: The store has not been resolved yet, so no engine exists.
        """
        if self._engine is None:  # pragma: no cover - reached only by calling out of order
            msg = "the database engine has not been opened; resolve the document store first"
            raise AssemblyError(msg)
        return self._engine

    async def connector(self, name: str) -> Connector:
        """A configured connector, by the instance name configuration gave it."""
        return await self._container.connector(name)


class _LazyParsers(Mapping[str, "Parser"]):
    """Every registered parser, constructed only when one is actually asked for.

    Citation verification needs *the* parser a document was parsed with, and nothing else. A
    mapping built eagerly would load pdfium, tree-sitter, python-docx, python-pptx, selectolax,
    nbformat and calamine to resolve one anchor into a Markdown file.
    """

    def __init__(self, container: Container) -> None:
        self._container = container

    @override
    def __getitem__(self, name: str) -> Parser:
        from manicule.core.errors import UnknownComponentError  # noqa: PLC0415

        try:
            return self._container.get(keys.PARSER.named(name))
        except UnknownComponentError as exc:
            raise KeyError(name) from exc

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._container.registry.names(ComponentKind.PARSER))

    @override
    def __len__(self) -> int:
        return len(self._container.registry.names(ComponentKind.PARSER))


@dataclass(frozen=True, slots=True)
class _ContainerChain:
    """The configured parser chain, read off the container rather than rebuilt beside it.

    Satisfies :class:`~manicule.generation.verification.ParserChainLike`. Building a second
    :class:`~manicule.parsers.chain.ParserChain` here would be a second answer to "which
    parser reads this media type", and a citation verified by the wrong reader is a citation
    certified against text the document never had.
    """

    container: Container

    @property
    def parsers(self) -> Mapping[str, Parser]:
        return _LazyParsers(self.container)

    def resolve(self, media_type: str) -> tuple[str, ...]:
        return tuple(self.container.parser_chain_names(media_type))


class _Ingestion:
    """The ingest operations a surface can start, over one runtime."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def index_path(
        self,
        path: Path,
        *,
        name: str,
        limit: int | None = None,
        force: bool = False,
        watching: Watching | None = None,
    ) -> RunReport:
        """Walk a path and ingest what is under it.

        The connector is constructed here rather than resolved from configuration, because the
        path is the argument. It is the **same class** the ``filesystem`` component builds, so
        a one-off index and a configured source cannot behave differently.
        """
        from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415

        connector = FilesystemConnector(path, name=name)
        pipeline = await self._runtime.pipeline()
        if force:
            mutation_guard = getattr(self._runtime, "derived_mutation_guard", None)
            if mutation_guard is None:  # protocol test doubles; Runtime always supplies it
                return await self._forced(connector, pipeline, limit=limit, watching=watching)
            async with mutation_guard():
                return await self._forced(connector, pipeline, limit=limit, watching=watching)
        return await pipeline.run(connector, limit=limit, watching=watching)

    async def _forced(
        self,
        connector: Connector,
        pipeline: IngestPipeline,
        *,
        limit: int | None,
        watching: Watching | None = None,
    ) -> RunReport:
        """Re-ingest every discovered document, skipping change detection.

        Separate from :meth:`~manicule.ingest.pipeline.IngestPipeline.run` rather than a flag
        on it, because forcing changes what a run *means*: the watermark must not advance from
        a pass that ignored change detection, and no ``--force`` should be able to make the
        next ordinary sync think it has already seen everything.
        """
        from manicule.ingest.pipeline import RunReport  # noqa: PLC0415

        report = RunReport(connector=connector.name)
        stream = connector.discover(None)
        try:
            async for discovered in stream:
                raw = await connector.fetch(discovered.ref)
                outcomes = await pipeline.ingest_raw(
                    raw,
                    source=connector.name,
                    version_token=discovered.version_token,
                    title=discovered.title,
                    force=True,
                )
                for position, outcome in enumerate(outcomes):
                    report.record(outcome, expanded=position > 0)
                if watching is not None:
                    watching(
                        f"{connector.name}: {report.indexed} of "
                        f"{report.discovered} discovered documents indexed"
                    )
                if limit is not None and report.discovered >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - an enumeration failure is not a crash
            if isinstance(exc, CapacityRefusedError):
                report.enumeration_completed = False
                report.refuse_capacity(exc)
            else:
                report.error_type = type(exc).__name__
                report.error_message = str(exc)
                report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
        return report

    async def sync(
        self,
        connector: str,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
        acquire_only: bool = False,
    ) -> RunReport:
        pipeline = (
            await self._runtime.acquisition_pipeline()
            if acquire_only
            else await self._runtime.pipeline()
        )
        return await pipeline.run(
            await self._runtime.connector(connector),
            limit=limit,
            watching=watching,
            acquire_only=acquire_only,
        )

    async def connector(self, name: str) -> Connector:
        """The connector :meth:`sync` would run, handed back instead of run.

        One line, and it is deliberately the same line :meth:`sync` opens with rather than a
        parallel construction beside it. The container caches per instance, so a caller that
        asks for a source's profiles and then syncs it gets one object and cannot observe two
        readings of one configuration.
        """
        return await self._runtime.connector(name)

    async def snapshot_status(self, connector: str) -> tuple[AcquisitionRun, bool] | None:
        """Read the active or newest promoted manifest from durable local identity."""
        from manicule.ingest.ports import AcquisitionStore  # noqa: PLC0415

        store = await self._runtime.documents()
        if not isinstance(store, AcquisitionStore):
            return None
        run = await store.latest_unsettled_acquisition_run(connector)
        if run is None:
            run = await store.latest_promoted_snapshot(connector, None)
        if run is None:
            return None
        return run, False

    async def snapshot_verify(self, run_id: str) -> tuple[AcquisitionRun, bool] | None:
        """Verify one workspace-owned manifest by opaque id, without reading source content."""
        from manicule.ingest.ports import AcquisitionStore  # noqa: PLC0415

        store = await self._runtime.documents()
        if not isinstance(store, AcquisitionStore):
            return None
        run = await store.get_acquisition_run(run_id)
        if run is None:
            return None
        return run, await store.verify_snapshot_manifest(run.id)

    def _rebuild_target(self):  # noqa: ANN202
        """Resolve the configured rebuild identity from declarations, never components."""
        import json  # noqa: PLC0415

        from manicule.core.embedding import EmbedFingerprint  # noqa: PLC0415
        from manicule.core.fingerprints import ChunkFingerprint  # noqa: PLC0415
        from manicule.core.rebuild import RebuildTarget  # noqa: PLC0415
        from manicule.ingest.glossary_lineage import glossary_fingerprint  # noqa: PLC0415
        from manicule.parsers.versions import parse_fingerprint  # noqa: PLC0415
        from manicule.plugins.registry import MiddlewareMetadata  # noqa: PLC0415
        from manicule.storage import models  # noqa: PLC0415

        chunk = self._runtime.container.metadata(keys.CHUNKER)
        embed = self._runtime.container.metadata(keys.EMBEDDER)
        if not isinstance(chunk, ChunkFingerprint):
            raise ManiculeError("configured chunker metadata did not declare a fingerprint")
        if not isinstance(embed, EmbedFingerprint):
            raise ManiculeError("configured embedder metadata did not declare a fingerprint")
        if chunk.tokenizer_id != embed.tokenizer_id:
            raise ManiculeError(
                "configured chunker metadata names a tokenizer different from the embedder"
            )
        if chunk.max_tokens > embed.max_sequence_length:
            raise ManiculeError(
                f'plugins.config."chunker.structural".max_tokens is {chunk.max_tokens}, '
                f"but the configured embedder reads at most {embed.max_sequence_length} "
                "tokens. Lower that setting or choose a model with a longer context window; "
                "silently clamping it would make the derived fingerprint false."
            )
        middleware: list[MiddlewareMetadata] = []
        for name in self._runtime.settings.plugins.middleware:
            declaration = self._runtime.container.metadata(keys.MIDDLEWARE.named(name))
            if not isinstance(declaration, MiddlewareMetadata):
                raise ManiculeError(
                    f"middleware {name!r} metadata did not declare its fingerprint behavior"
                )
            middleware.append(declaration)
        chain = tuple(sorted(f"{item.name}@" for item in middleware))
        embedded = tuple(
            sorted(f"{item.name}@" for item in middleware if item.mutates_embedded_text)
        )
        chunk = chunk.with_middleware(embedded)
        settings = self._runtime.settings
        parser_names = tuple(self._runtime.container.registry.names(ComponentKind.PARSER))
        parser_set = tuple(
            fingerprint.canonical()
            for name in parser_names
            if (fingerprint := parse_fingerprint(name)) is not None
        )
        routing = hashlib.sha256(
            json.dumps(
                {"fallbacks": settings.parser_fallbacks, "parsers": parser_names},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        glossary = glossary_fingerprint(
            enabled=settings.rag.glossary.detect_on_ingest,
            middleware=chain,
        )
        return RebuildTarget(
            parser_routing=routing,
            parser_set=parser_set,
            chunk_fingerprint=chunk.canonical(),
            embedding_fingerprint=embed.canonical(),
            embedding_config=embed.model_dump_json(),
            glossary_fingerprint=glossary.canonical(),
            fts_tokenizer=models.FTS_TOKENIZER,
            batch_documents=max(1, min(32, settings.ingest.max_documents_per_worker)),
            max_memory_bytes=settings.ingest.parse_memory_limit_mb * 1024 * 1024,
            max_temporary_bytes=settings.ingest.max_acquired_blob_backlog_bytes,
        )

    async def _rebuild_components(self, snapshot_run_id: str):  # noqa: ANN202
        """Assemble the connector-free production offline rebuild stack and exact target."""
        from manicule.ingest.rebuild import build_offline_rebuilder  # noqa: PLC0415
        from manicule.ingest.workers import WorkerPool, worker_config  # noqa: PLC0415
        from manicule.parsers.chain import ParserChain  # noqa: PLC0415
        from manicule.storage.blobs import BlobStore  # noqa: PLC0415
        from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
        from manicule.storage.rebuild import SqliteRebuildStore  # noqa: PLC0415

        documents = await self._runtime.documents()
        if not isinstance(documents, SqliteDocStore):
            raise ManiculeError("offline rebuild requires the built-in SQLite document store")
        snapshot = await documents.get_acquisition_run(snapshot_run_id)
        if snapshot is None:
            raise UnknownEntityError("no durable source snapshot has that id")
        blobs = await self._runtime.blobs()
        if not isinstance(blobs, BlobStore):
            raise ManiculeError("offline rebuild requires retained local source bytes")
        vectors = await self._runtime.prepared_vectors()
        store = SqliteRebuildStore(
            self._runtime.require_engine(),
            workspace_id=self._runtime.workspace,
            blobs=blobs,
            vectors=vectors,  # pyright: ignore[reportArgumentType]
        )
        chunker = await self._runtime.container.aget(keys.CHUNKER)
        embedder = await self._runtime.container.aget(keys.EMBEDDER)
        middleware = await self._runtime.middleware()
        chunk_fingerprint = chunker.fingerprint.with_middleware(middleware.declarations())
        settings = self._runtime.settings
        target = self._rebuild_target()
        routing = target.parser_routing
        actual_glossary = await self.glossary_fingerprint()
        mismatches: list[str] = []
        if (
            target.embedding_fingerprint != embedder.fingerprint.canonical()
            or target.embedding_config != embedder.fingerprint.model_dump_json()
        ):
            mismatches.append("embedder")
        if target.chunk_fingerprint != chunk_fingerprint.canonical():
            mismatches.append("chunker or middleware")
        if target.glossary_fingerprint != actual_glossary.canonical():
            mismatches.append("glossary middleware")
        if mismatches:
            raise ManiculeError(
                "component metadata disagrees with the executable rebuild stack: "
                + ", ".join(mismatches)
            )
        runner = WorkerPool(
            worker_config(settings, chunker=chunker, embedder=embedder),
            workers=settings.ingest.parse_workers,
            timeout_s=settings.ingest.parse_timeout_s,
            poll_interval_s=settings.ingest.memory_poll_interval_s,
            max_documents=settings.ingest.max_documents_per_worker,
        )
        rebuilder = build_offline_rebuilder(
            store=store,
            blobs=blobs,
            workspace_id=self._runtime.workspace,
            source=snapshot.connector,
            parser_chain=cast("ParserChain", _ContainerChain(self._runtime.container)),
            routing_identity=routing,
            chunker=chunker,
            embedder=embedder,
            vectors=vectors,
            chunk_fingerprint=chunk_fingerprint,
            middleware=middleware,
            parse_runner=runner,
            detect_glossary=settings.rag.glossary.detect_on_ingest,
        )
        return store, rebuilder, target

    async def rebuild_plan(self, snapshot_run_id: str):  # noqa: ANN202
        from manicule.ingest.rebuild import MAX_MISSING_DETAILS  # noqa: PLC0415
        from manicule.storage.blobs import BlobStore  # noqa: PLC0415
        from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415
        from manicule.storage.rebuild import SqliteRebuildStore  # noqa: PLC0415

        documents = await self._runtime.documents()
        if not isinstance(documents, SqliteDocStore):
            raise ManiculeError("offline rebuild requires the built-in SQLite document store")
        if await documents.get_acquisition_run(snapshot_run_id) is None:
            raise UnknownEntityError("no durable source snapshot has that id")
        blobs = await self._runtime.blobs()
        if not isinstance(blobs, BlobStore):
            raise ManiculeError("offline rebuild requires retained local source bytes")
        store = SqliteRebuildStore(
            self._runtime.require_engine(),
            workspace_id=self._runtime.workspace,
            blobs=blobs,
        )
        return await store.plan_rebuild(
            snapshot_run_id,
            self._rebuild_target(),
            missing_limit=MAX_MISSING_DETAILS,
            persist=False,
        )

    async def rebuild_run(self, snapshot_run_id: str, owner: str):  # noqa: ANN202
        async with self._runtime.derived_mutation_guard():
            return await self._rebuild_run_guarded(snapshot_run_id, owner)

    async def _rebuild_run_guarded(self, snapshot_run_id: str, owner: str):  # noqa: ANN202
        _, rebuilder, target = await self._rebuild_components(snapshot_run_id)
        return await rebuilder.run(snapshot_run_id, target, owner=owner)

    async def rebuild_status(self, generation_id: str):  # noqa: ANN202
        from manicule.storage.rebuild import SqliteRebuildStore  # noqa: PLC0415

        blobs = await self._runtime.blobs()
        vectors = await self._runtime.vectors()
        store = SqliteRebuildStore(
            self._runtime.require_engine(),
            workspace_id=self._runtime.workspace,
            blobs=blobs,  # pyright: ignore[reportArgumentType]
            vectors=vectors,  # pyright: ignore[reportArgumentType]
        )
        try:
            return await store.checkpoint(generation_id)
        except KeyError:
            return None

    async def reembed_plan(self) -> tuple[ReembedPlan, str, int]:
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_plan_guarded()

    async def _reembed_plan_guarded(self) -> tuple[ReembedPlan, str, int]:
        """Price from a transient durable snapshot, with no embedding or source access."""
        from manicule.ingest.reembed import (  # noqa: PLC0415
            discard_reembed_snapshot,
            plan_reembed_commitment,
        )
        from manicule.storage.reembed import SqliteReembedCorpus  # noqa: PLC0415

        await self._require_reembed_backend()
        await self._runtime.documents()
        corpus = SqliteReembedCorpus(self._runtime.require_engine(), self._runtime.workspace)
        embedder = await self._runtime.embedder()
        commitment = await plan_reembed_commitment(corpus, embedder.fingerprint)
        try:
            return (
                commitment.plan,
                hashlib.sha256(commitment.target_fingerprint.encode("utf-8")).hexdigest(),
                commitment.target_dimension,
            )
        finally:
            await discard_reembed_snapshot(corpus, commitment.snapshot.id)

    async def reembed_start(self, run_id: str, owner_token: str) -> ReembedRun:
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_start_guarded(run_id, owner_token)

    async def _reembed_start_guarded(self, run_id: str, owner_token: str) -> ReembedRun:
        from manicule.ingest.reembed import (  # noqa: PLC0415
            ReembedCapacityError,
            discard_reembed_snapshot,
            plan_reembed_commitment,
            start_reembed,
        )
        from manicule.storage.reembed import SqliteReembedStore  # noqa: PLC0415

        await self._require_reembed_backend()
        await self._runtime.documents()
        authority = SqliteReembedStore(self._runtime.require_engine(), self._runtime.workspace)
        existing = await authority.get(run_id)
        if existing is not None:
            return existing
        corpus, authority, _shadows, embedder = await self._reembed_components()
        commitment = await plan_reembed_commitment(corpus, embedder.fingerprint)
        if not commitment.execution_plan.runnable:
            error = PolicyError(
                "one or more stored documents have no chunks. "
                "Re-embedding never falls back to a connector or parser; repair local inputs "
                "first."
            )
            await discard_reembed_snapshot(corpus, commitment.snapshot.id, error)
            raise error
        try:
            self._require_reembed_capacity_bytes(commitment.execution_plan.temporary_disk_bytes)
        except ReembedCapacityError as error:
            await discard_reembed_snapshot(corpus, commitment.snapshot.id, error)
            raise
        # Starting is deliberately a durable, non-embedding checkpoint.  The operator receives
        # the run id before the expensive step begins, so a process exit can always be recovered
        # with ``resume`` rather than losing the only handle to the journal row.
        try:
            return await start_reembed(
                run_id,
                owner_token=owner_token,
                corpus=corpus,
                target=embedder.fingerprint,
                journal=authority,
                commitment=commitment,
            )
        except BaseException as error:
            await discard_reembed_snapshot(corpus, commitment.snapshot.id, error)
            raise

    async def reembed_resume(self, run_id: str, owner_token: str) -> ReembedRun:
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_resume_guarded(run_id, owner_token)

    async def _reembed_resume_guarded(self, run_id: str, owner_token: str) -> ReembedRun:
        from manicule.storage.reembed import (  # noqa: PLC0415
            LanceShadowGenerations,
            SqliteReembedCorpus,
            SqliteReembedStore,
        )

        await self._require_reembed_backend()
        await self._runtime.documents()
        engine = self._runtime.require_engine()
        authority = SqliteReembedStore(engine, self._runtime.workspace)
        run = await authority.get(run_id)
        if run is None:
            raise UnknownEntityError("no durable re-embedding run has that id")
        # Refuse for local disk before constructing a potentially multi-gigabyte model runtime.
        self._require_reembed_capacity(run)
        corpus = SqliteReembedCorpus(engine, self._runtime.workspace)
        shadows = LanceShadowGenerations(await self._runtime.vector_directory(), authority)
        embedder = await self._runtime.embedder()
        return await self._resume_reembed(
            run_id=run_id,
            owner_token=owner_token,
            corpus=corpus,
            authority=authority,
            shadows=shadows,
            embedder=embedder,
        )

    async def reembed_status(self, run_id: str) -> ReembedRun | None:
        from manicule.storage.reembed import SqliteReembedStore  # noqa: PLC0415

        await self._runtime.documents()
        return await SqliteReembedStore(
            self._runtime.require_engine(), self._runtime.workspace
        ).get(run_id)

    async def reembed_abandon(self, run_id: str, owner_token: str) -> ReembedRun:
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_abandon_guarded(run_id, owner_token)

    async def _reembed_abandon_guarded(self, run_id: str, owner_token: str) -> ReembedRun:
        from manicule.storage.reembed import SqliteReembedStore  # noqa: PLC0415

        await self._runtime.documents()
        authority = SqliteReembedStore(self._runtime.require_engine(), self._runtime.workspace)
        if await authority.get(run_id) is None:
            raise UnknownEntityError("no durable re-embedding run has that id")
        lease = await authority.acquire(run_id, owner_token, ttl_seconds=30.0)
        return await authority.abandon(run_id, lease=lease)

    async def reembed_cleanup(self, run_id: str) -> bool:
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_cleanup_guarded(run_id)

    async def _reembed_cleanup_guarded(self, run_id: str) -> bool:
        from manicule.storage.reembed import (  # noqa: PLC0415
            LanceShadowGenerations,
            SqliteReembedStore,
        )

        await self._require_reembed_backend()
        await self._runtime.documents()
        authority = SqliteReembedStore(self._runtime.require_engine(), self._runtime.workspace)
        return await LanceShadowGenerations(
            await self._runtime.vector_directory(), authority
        ).cleanup_terminal(run_id)

    async def reembed_recover_pending(self) -> ReembedRecovery:
        """Resume ownerless runs after a serving-process restart, scoped to this workspace."""
        async with self._runtime.derived_mutation_guard():
            return await self._reembed_recover_pending_guarded()

    async def _reembed_recover_pending_guarded(self) -> ReembedRecovery:
        """Recover while reset cannot terminalize a run between listing and resumption."""
        import secrets  # noqa: PLC0415

        from manicule.storage.reembed import SqliteReembedStore  # noqa: PLC0415

        await self._require_reembed_backend()
        await self._runtime.documents()
        authority = SqliteReembedStore(self._runtime.require_engine(), self._runtime.workspace)
        return await _recover_reembed_runs(
            await authority.recoverable_run_ids(),
            lambda run_id: self.reembed_resume(run_id, secrets.token_urlsafe(24)),
        )

    async def _reembed_components(
        self,
    ) -> tuple[SqliteReembedCorpus, SqliteReembedStore, LanceShadowGenerations, Embedder]:
        from manicule.storage.reembed import (  # noqa: PLC0415
            LanceShadowGenerations,
            SqliteReembedCorpus,
            SqliteReembedStore,
        )

        await self._runtime.documents()
        engine = self._runtime.require_engine()
        authority = SqliteReembedStore(engine, self._runtime.workspace)
        return (
            SqliteReembedCorpus(engine, self._runtime.workspace),
            authority,
            LanceShadowGenerations(await self._runtime.vector_directory(), authority),
            await self._runtime.embedder(),
        )

    async def _require_reembed_backend(self) -> None:
        from manicule.ingest.reembed import ReembedError  # noqa: PLC0415
        from manicule.storage.vectors import PublishedLanceVectorStore  # noqa: PLC0415

        vectors = await self._runtime.vectors()
        if not isinstance(vectors, PublishedLanceVectorStore):
            raise ReembedError(
                "durable re-embedding requires the built-in SQLite/Lance vector backend; "
                "the configured custom backend does not implement named shadow generations, "
                "atomic publication, inspection, and cleanup"
            )

    def _require_reembed_capacity(self, run: ReembedRun) -> None:
        self._require_reembed_capacity_bytes(run.commitment.execution_plan.temporary_disk_bytes)

    def _require_reembed_capacity_bytes(self, required: int) -> None:
        from manicule.ingest.reembed import ReembedCapacityError  # noqa: PLC0415

        available = shutil.disk_usage(self._runtime.settings.data_dir).free
        if required > available:
            raise ReembedCapacityError(
                "re-embedding does not have enough local temporary disk capacity; free local "
                "disk space and retry (resume an existing durable run with the same id)"
            )

    async def _resume_reembed(
        self,
        *,
        run_id: str,
        owner_token: str,
        corpus: SqliteReembedCorpus,
        authority: SqliteReembedStore,
        shadows: LanceShadowGenerations,
        embedder: Embedder,
    ) -> ReembedRun:
        from manicule.ingest.reembed import ReembedState, resume_reembed  # noqa: PLC0415

        run = await resume_reembed(
            run_id,
            owner_token=owner_token,
            corpus=corpus,
            embedder=embedder,
            journal=authority,
            shadow=shadows,
            publisher=authority,
        )
        if run.state is ReembedState.PUBLISHED:
            vectors = await self._runtime.vectors()
            await vectors.ensure_ready(embedder.fingerprint)
        return run

    async def reindex(self, document_id: str) -> ReindexReport:
        async with self._runtime.derived_mutation_guard():
            return await self._reindex_guarded(document_id)

    async def _reindex_guarded(self, document_id: str) -> ReindexReport:
        from manicule.ingest.reindex import reindex_document  # noqa: PLC0415

        store = await self._runtime.documents()
        return await reindex_document(
            document_id,
            store=store,  # pyright: ignore[reportArgumentType] - the store satisfies IngestStore
            pipeline=await self._runtime.pipeline(),
            blobs=await self._runtime.blobs(),
        )

    async def reparse_stale(self, *, batch: int, dry_run: bool = False) -> StaleSweep:
        if dry_run:
            return await self._reparse_stale_guarded(batch=batch, dry_run=True)
        async with self._runtime.derived_mutation_guard():
            return await self._reparse_stale_guarded(batch=batch, dry_run=False)

    async def _reparse_stale_guarded(self, *, batch: int, dry_run: bool = False) -> StaleSweep:
        """Sweep the corpus for documents an installed parser has moved past.

        ``current_parse_fingerprints`` is read here, once per run, and it *raises* when a
        declared distribution is missing. That is the right place for it to raise: a partial
        set of current fingerprints makes every document the absent parser produced look
        stale, and a sweep that reparsed them would rebuild a corpus with a parser that is not
        there. Failing before the first document beats reporting a repair that could not have
        happened.
        """
        from manicule.ingest.reindex import plan_stale, re_parse_stale  # noqa: PLC0415
        from manicule.parsers.versions import current_parse_fingerprints  # noqa: PLC0415

        store = await self._runtime.documents()
        current = current_parse_fingerprints()
        if dry_run:
            # Before the pipeline, deliberately. Building one constructs a chunker, an
            # embedder, a vector store and a pool of parse workers, and then refuses outright
            # if the index disagrees with any of them — which is right for a run that is about
            # to write and wrong for a survey of what it would touch. A plan reads rows.
            return await plan_stale(
                store=store,  # pyright: ignore[reportArgumentType] - it satisfies IngestStore
                parse_fingerprints=current,
                batch=batch,
            )
        return await re_parse_stale(
            store=store,  # pyright: ignore[reportArgumentType] - the store satisfies IngestStore
            pipeline=await self._runtime.pipeline(),
            blobs=await self._runtime.blobs(),
            parse_fingerprints=current,
            batch=batch,
        )

    async def glossary_fingerprint(self) -> GlossaryFingerprint:
        """What the installed detector would produce under this configuration.

        The single reader of the two inputs, and the reason it is a method rather than three
        lines repeated at four call sites: ``status``, ``doctor``, the repair and the pipeline
        all need this string, and a second reading of the middleware chain in any one of them
        would report a corpus stale that the others called current.

        Builds nothing. The container is asked for its hooks, which are already-constructed
        objects, and the digest is read off two source files.
        """
        from manicule.ingest.glossary_lineage import glossary_fingerprint  # noqa: PLC0415

        middleware = await self._runtime.middleware()
        return glossary_fingerprint(
            enabled=self._runtime.settings.rag.glossary.detect_on_ingest,
            middleware=middleware.chain(),
        )

    async def configured_index_fingerprints(self) -> tuple[str, str]:
        """Return declaration-only target identities; do not construct an embedder or parser."""
        target = self._rebuild_target()
        return target.embedding_fingerprint, target.chunk_fingerprint

    async def physical_index_fingerprint(self) -> str | None:
        """Inspect only existing vector metadata; never prepare a fresh table."""
        vectors = await self._runtime.vectors()
        inspect = getattr(vectors, "physical_fingerprint", None)
        if inspect is None:
            return None
        fingerprint = await inspect()
        return None if fingerprint is None else fingerprint.canonical()

    async def redetect_stale_glossary(self, *, batch: int, dry_run: bool = False) -> GlossarySweep:
        if dry_run:
            return await self._redetect_stale_glossary_guarded(batch=batch, dry_run=True)
        async with self._runtime.derived_mutation_guard():
            return await self._redetect_stale_glossary_guarded(batch=batch, dry_run=False)

    async def _redetect_stale_glossary_guarded(
        self, *, batch: int, dry_run: bool = False
    ) -> GlossarySweep:
        """Bring every document's glossary up to the installed detector. Reads chunks only.

        **No pipeline is built on either path, and that is the cost boundary rather than a
        convenience.** :meth:`reparse_stale` constructs one for its real run because it parses;
        this never does, so it never constructs a chunker, an embedder, a vector store or a
        pool of parse workers, and it never opens the blob store. What it needs is the document
        store, twice — once as the thing that answers the selection and reads chunks, once as
        the thing that holds entries — and the same object is both.

        The consequence worth stating: the fingerprint refusals that guard a writing run are not
        run here, deliberately. They exist because a corpus must not be *written* to by a
        different chunker or embedder, and this writes neither chunks nor vectors. An index
        whose embedder disagrees with configuration is one an operator must still be able to
        bring on to current detection rules.
        """
        from manicule.ingest.reindex import (  # noqa: PLC0415
            plan_stale_glossary,
            redetect_stale_glossary,
        )

        store = await self._runtime.documents()
        fingerprint = await self.glossary_fingerprint()
        if dry_run:
            return await plan_stale_glossary(
                store=store,  # pyright: ignore[reportArgumentType] - it satisfies IngestStore
                fingerprint=fingerprint,
                batch=batch,
            )
        return await redetect_stale_glossary(
            store=store,  # pyright: ignore[reportArgumentType] - it satisfies IngestStore
            glossary=store,  # pyright: ignore[reportArgumentType] - and GlossaryStore
            fingerprint=fingerprint,
            batch=batch,
        )

    async def import_archive(self, path: Path, *, force: bool = False) -> RunReport:
        mutation_guard = getattr(self._runtime, "derived_mutation_guard", None)
        if mutation_guard is None:  # protocol test doubles; Runtime always supplies it
            return await self._import_archive_guarded(path, force=force)
        async with mutation_guard():
            return await self._import_archive_guarded(path, force=force)

    async def _import_archive_guarded(self, path: Path, *, force: bool = False) -> RunReport:
        """Ingest an exported archive through the ordinary pipeline.

        Chunks and vectors are produced **here**, by this installation's chunker and embedder.
        The archive carries source bytes and metadata and nothing derived, so an import can
        never introduce an index built by something else.
        """
        from manicule.ingest.pipeline import RunReport  # noqa: PLC0415

        manifest_path, manifest = await asyncio.to_thread(_open_archive, path)
        root = manifest_path.parent
        pipeline = await self._runtime.pipeline()
        report = RunReport(connector="import")
        for entry in manifest.documents:
            blob = root / entry.blob
            content = await asyncio.to_thread(_read_blob, blob)
            if content is None:
                report.error = f"archive entry {entry.source_id!r} has no bytes at {blob}"
                continue
            raw = RawDocument(
                source_id=entry.source_id,
                uri=entry.uri,
                media_type=entry.media_type,
                content=content,
            )
            try:
                outcomes = await pipeline.ingest_raw(
                    raw,
                    source=entry.source,
                    version_token=entry.version_token,
                    title=entry.title,
                    force=force,
                )
            except CapacityRefusedError as error:
                report.enumeration_completed = False
                report.refuse_capacity(error)
                break
            for position, outcome in enumerate(outcomes):
                report.record(outcome, expanded=position > 0)
        return report


class _Keys:
    """API keys, over one runtime's engine and workspace.

    The secret is generated here, hashed here, and returned exactly once. Only the digest
    reaches the database, so a copy of the database — a backup, an export, a support bundle —
    is not a copy of the credentials.
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def issue(
        self, name: str, *, role: str, expires_days: int | None = None
    ) -> tuple[ApiKeySummary, str]:
        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415
        from manicule.storage.types import utcnow  # noqa: PLC0415

        secret = f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        created = utcnow()
        expires = created + timedelta(days=expires_days) if expires_days else None
        row = models.ApiKey(
            id=secrets.token_hex(8),
            name=name,
            key_hash=digest,
            key_prefix=secret[: len(KEY_PREFIX) + 6],
            workspace_id=self._runtime.workspace,
            user_id=self._runtime.workspace,
            role=role,
            scopes=[],
            allowed_ips=[],
            expires_at=expires,
            created_at=created,
        )
        sessions = session_factory(self._runtime.require_engine())
        async with sessions.begin() as session:
            session.add(row)
        return _key_summary(row), secret

    async def list_keys(self) -> Sequence[ApiKeySummary]:
        from sqlalchemy import select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        sessions = session_factory(self._runtime.require_engine())
        async with sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.ApiKey)
                        .where(models.ApiKey.workspace_id == self._runtime.workspace)
                        .order_by(models.ApiKey.created_at)
                    )
                )
                .scalars()
                .all()
            )
        return [_key_summary(row) for row in rows]

    async def verify(self, secret: str) -> ApiKeySummary | None:
        """Resolve a presented secret to its key, or ``None``.

        Four predicates, all in the statement: the digest matches, the key belongs to **this**
        workspace, it has not been revoked, and it has not expired. The digest is what is
        compared — the plaintext is never stored — so there is no string comparison here to
        time a byte at a time, and no branch that treats an unknown key differently from a
        revoked one.

        ``last_used_at`` is deliberately **not** updated. It would turn every authenticated
        read into a write, serializing the whole API behind SQLite's single writer, and
        "when was this key last used" is a question the audit trail answers without that cost.
        """
        from sqlalchemy import or_, select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415
        from manicule.storage.types import utcnow  # noqa: PLC0415

        if not secret:
            return None
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        now = utcnow()
        sessions = session_factory(self._runtime.require_engine())
        async with sessions() as session:
            row = (
                (
                    await session.execute(
                        select(models.ApiKey).where(
                            models.ApiKey.key_hash == digest,
                            models.ApiKey.workspace_id == self._runtime.workspace,
                            models.ApiKey.revoked_at.is_(None),
                            or_(
                                models.ApiKey.expires_at.is_(None),
                                models.ApiKey.expires_at > now,
                            ),
                        )
                    )
                )
                .scalars()
                .first()
            )
        return None if row is None else _key_summary(row)

    async def revoke(self, name_or_id: str) -> ApiKeySummary:
        from sqlalchemy import or_, select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415
        from manicule.storage.types import utcnow  # noqa: PLC0415

        sessions = session_factory(self._runtime.require_engine())
        async with sessions.begin() as session:
            row = (
                (
                    await session.execute(
                        select(models.ApiKey).where(
                            # Scoped to this workspace, and not as a courtesy: a revoke that
                            # could reach another tenant's key is a denial-of-service across
                            # the boundary the whole design exists to hold.
                            models.ApiKey.workspace_id == self._runtime.workspace,
                            or_(models.ApiKey.id == name_or_id, models.ApiKey.name == name_or_id),
                        )
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                msg = f"no API key named {name_or_id!r} in workspace {self._runtime.workspace!r}"
                raise UnknownEntityError(msg)
            row.revoked_at = utcnow()
            return _key_summary(row)


class _Telemetry:
    """Query logs and the audit trail, over one runtime's engine and workspace.

    The two tables differ in one way that matters and is easy to get backwards.
    ``query_logs.workspace_id`` is a foreign key that cascades — query text is user content,
    and keeping it past its workspace's deletion is a retention problem. ``audit_logs`` has
    **no** foreign keys at all: an audit trail that cascades away with the thing it audits is
    not an audit trail. So one is filtered by a joinable column and the other by a plain one,
    and neither is written the way the other is.
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def record_query(
        self,
        query: str,
        *,
        profile: str,
        chunk_ids: Sequence[str],
        confidence: float | None,
        elapsed_ms: int,
    ) -> str:
        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        identifier = secrets.token_hex(8)
        sessions = session_factory(self._runtime.require_engine())
        async with sessions.begin() as session:
            session.add(
                models.QueryLog(
                    id=identifier,
                    workspace_id=self._runtime.workspace,
                    query=query,
                    profile=profile,
                    retrieved_chunk_ids=list(chunk_ids),
                    confidence_score=confidence,
                    response_time_ms=elapsed_ms,
                )
            )
        return identifier

    async def query_logs(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        from sqlalchemy import func, select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        sessions = session_factory(self._runtime.require_engine())
        scope = models.QueryLog.workspace_id == self._runtime.workspace
        async with sessions() as session:
            total = (
                await session.execute(select(func.count(models.QueryLog.id)).where(scope))
            ).scalar_one()
            rows = (
                (
                    await session.execute(
                        select(models.QueryLog)
                        .where(scope)
                        .order_by(models.QueryLog.created_at.desc(), models.QueryLog.id)
                        .limit(max(limit, 0))
                        .offset(max(offset, 0))
                    )
                )
                .scalars()
                .all()
            )
        return [
            {
                "id": row.id,
                "query": row.query,
                "profile": row.profile or "",
                # The count, not the ids. A page of telemetry is read by a person looking for
                # slow or low-confidence queries, and a list of chunk ids is corpus structure
                # that has no business traveling with it.
                "chunks": _length(row.retrieved_chunk_ids),
                "confidence": row.confidence_score,
                "elapsed_ms": row.response_time_ms,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ], int(total)

    async def record_audit(
        self,
        event_type: str,
        *,
        details: Mapping[str, object],
        actor: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        sessions = session_factory(self._runtime.require_engine())
        async with sessions.begin() as session:
            session.add(
                models.AuditLog(
                    id=secrets.token_hex(8),
                    workspace_id=self._runtime.workspace,
                    user_id=actor,
                    event_type=event_type,
                    details=dict(details),
                    ip_address=ip_address,
                )
            )

    async def audit_logs(
        self, *, limit: int = 50, offset: int = 0, event_type: str | None = None
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        from sqlalchemy import func, select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        sessions = session_factory(self._runtime.require_engine())
        clauses = [models.AuditLog.workspace_id == self._runtime.workspace]
        if event_type:
            clauses.append(models.AuditLog.event_type == event_type)
        async with sessions() as session:
            total = (
                await session.execute(select(func.count(models.AuditLog.id)).where(*clauses))
            ).scalar_one()
            rows = (
                (
                    await session.execute(
                        select(models.AuditLog)
                        .where(*clauses)
                        .order_by(models.AuditLog.created_at.desc(), models.AuditLog.id)
                        .limit(max(limit, 0))
                        .offset(max(offset, 0))
                    )
                )
                .scalars()
                .all()
            )
        return [
            {
                "id": row.id,
                "event_type": row.event_type,
                "actor": row.user_id,
                "ip_address": row.ip_address,
                "details": row.details if isinstance(row.details, dict) else {},
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ], int(total)


class _Maintenance:
    """Whole-installation operations, over one runtime's engine and data directory."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def schema_revision(self) -> str | None:
        from manicule.storage.migrator import current  # noqa: PLC0415

        return await current(self._runtime.require_engine())

    async def backup(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> Mapping[str, object]:
        from manicule.storage.backup import create_backup  # noqa: PLC0415

        settings = self._runtime.settings
        return await create_backup(
            self._runtime.require_engine(),
            settings.data_dir,
            target,
            allow_insecure_target=allow_insecure_target,
        )

    async def restore(self, source: Path, *, force: bool = False) -> Mapping[str, object]:
        from manicule.storage.backup import restore_backup  # noqa: PLC0415

        settings = self._runtime.settings
        # Disposed first, deliberately. Copying a database file out from under an open pool
        # leaves connections pointing at an inode that no longer exists, and the next query
        # reads the *old* file with no error raised.
        await self._runtime.aclose()
        return restore_backup(source, settings.data_dir, force=force)

    async def reset_index(self) -> ResetOutcome:
        """Delete derived chunks and vectors while retaining source manifests and documents.

        Relational visibility is retired first and every unfinished lease/checkpoint is fenced.
        Physical publications are then removed by their durable generation/table bindings. The
        owned cleanup is joined through caller cancellation, so return or cancellation both leave
        the reset at a durable terminal or retryable boundary.
        """

        async with self._runtime.derived_mutation_guard():
            lifecycle = self._source_lifecycle(await self._runtime.documents())
            return await self._reset_derived_joined(lifecycle)

    async def plan_reset_derived(self) -> LifecyclePlan:
        return await self._source_lifecycle(await self._runtime.documents()).plan_reset_derived()

    async def reset_derived(self) -> LifecycleOutcome:
        from manicule.core.source_lifecycle import (  # noqa: PLC0415
            LifecycleOperation,
            LifecycleOutcome,
        )

        lifecycle = self._source_lifecycle(await self._runtime.documents())
        async with self._runtime.derived_mutation_guard():
            outcome = await self._reset_derived_joined(lifecycle)
        return LifecycleOutcome(
            operation=LifecycleOperation.RESET_DERIVED,
            removed_items=outcome.chunks,
            snapshot_items=outcome.snapshots_retained,
        )

    async def plan_derived_generation_cleanup(self) -> LifecyclePlan:
        return await self._source_lifecycle(
            await self._runtime.documents()
        ).plan_derived_generation_cleanup()

    async def cleanup_derived_generations(self) -> LifecycleOutcome:
        async with self._runtime.derived_mutation_guard():
            return await self._cleanup_derived_generations_guarded()

    async def _cleanup_derived_generations_guarded(self) -> LifecycleOutcome:
        lifecycle = self._source_lifecycle(await self._runtime.documents())
        vectors = await self._runtime.vectors()
        remover = getattr(vectors, "delete_bound_publication", None)
        if remover is None:
            raise ManiculeError(
                "derived-generation cleanup requires a vector backend with bound publications"
            )
        removed = 0
        released = 0
        async for page in lifecycle.obsolete_generation_publications():
            for generation in page:
                if generation.vector_publication_id is not None:
                    await remover(
                        generation.expected_vector_table,
                        generation.vector_publication_id,
                    )
                count, bytes_ = await lifecycle.cleanup_obsolete_generation(
                    generation.generation_id
                )
                removed += count
                released += bytes_
        from manicule.core.source_lifecycle import (  # noqa: PLC0415
            LifecycleOperation,
            LifecycleOutcome,
        )

        return LifecycleOutcome(
            operation=LifecycleOperation.CLEANUP_DERIVED_GENERATIONS,
            removed_items=removed,
            released_bytes=released,
        )

    async def _reset_derived_joined(self, lifecycle: SqliteDocStore) -> ResetOutcome:
        """Commit visibility first, then join owned physical cleanup through cancellation."""
        import asyncio  # noqa: PLC0415 - only this lifecycle boundary owns a task

        async def settle():  # noqa: ANN202
            from manicule.app.ports import ResetOutcome  # noqa: PLC0415
            from manicule.storage.vectors import (  # noqa: PLC0415
                PublishedLanceVectorStore,
                generation_pin,
                reset_vector_directory,
            )

            directory = await self._runtime.vector_directory()
            async with generation_pin(directory, exclusive=True):
                prepared = await lifecycle.prepare_reset_derived()
                vectors = await self._runtime.vectors()
                if not isinstance(vectors, PublishedLanceVectorStore):
                    raise ManiculeError(
                        "derived reset requires the built-in publication-aware vector backend"
                    )
                vector_rows = 0
                while tombstones := await lifecycle.reset_vector_tombstones():
                    grouped: dict[str | None, list[str]] = {}
                    for tombstone in tombstones:
                        grouped.setdefault(tombstone.vector_table, []).append(tombstone.vector_id)
                    for vector_table, vector_ids in grouped.items():
                        vector_rows += await vectors.delete_bound_chunks(vector_table, vector_ids)
                        await lifecycle.clear_tombstones(vector_ids)
                cleanup = await self.cleanup_derived_generations()
                await lifecycle.retire_reset_pointer()
                from manicule.storage.reembed import (  # noqa: PLC0415
                    LanceShadowGenerations,
                    SqliteReembedStore,
                )

                authority = SqliteReembedStore(
                    self._runtime.require_engine(), self._runtime.workspace
                )
                shadows = LanceShadowGenerations(directory, authority)
                for run_id in await lifecycle.reset_reembed_run_ids():
                    await shadows.cleanup_terminal(run_id)
                other_legacy = await lifecycle.other_legacy_vector_consumers()
                remove_store = prepared.vector_namespace != "legacy" or other_legacy == 0
                physical_removed = False
                if remove_store:
                    await vectors.teardown()
                    physical_removed = await reset_vector_directory(
                        directory, legacy_root=prepared.vector_namespace == "legacy"
                    )
                    if prepared.vector_namespace == "legacy":
                        vector_rows += await lifecycle.clear_unowned_legacy_tombstones()
                fingerprints = await lifecycle.finish_reset_identity()
                await self._runtime.invalidate_derived_runtime()
                return ResetOutcome(
                    documents=prepared.documents,
                    chunks=prepared.chunks,
                    memberships=prepared.memberships,
                    vector_rows=vector_rows,
                    publications=cleanup.removed_items,
                    generations_terminalized=prepared.generations_terminalized,
                    snapshots_retained=prepared.snapshots,
                    vector_store_removed=physical_removed,
                    fingerprints_cleared=fingerprints,
                    runtime_cache_invalidated=True,
                )

        work = asyncio.create_task(settle(), name="lifecycle:reset-derived")
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(work)
            except asyncio.CancelledError:
                if work.cancelled():
                    raise
                interrupted = True
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
            else:
                if interrupted:
                    raise asyncio.CancelledError
                return result

    async def plan_source_history_release(self, cutoff: datetime) -> LifecyclePlan:
        return await self._source_lifecycle(
            await self._runtime.documents()
        ).plan_source_history_release(cutoff)

    async def release_source_history(self, cutoff: datetime) -> LifecycleOutcome:
        return await self._source_lifecycle(await self._runtime.documents()).release_source_history(
            cutoff
        )

    async def plan_snapshot_deletion(self, run_id: str) -> LifecyclePlan:
        return await self._source_lifecycle(await self._runtime.documents()).plan_snapshot_deletion(
            run_id
        )

    async def delete_snapshot(self, run_id: str, *, confirmation: str) -> LifecycleOutcome:
        return await self._source_lifecycle(await self._runtime.documents()).delete_snapshot(
            run_id, confirmation=confirmation
        )

    @staticmethod
    def _source_lifecycle(store: DocumentSurface) -> SqliteDocStore:
        from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

        if not isinstance(store, SqliteDocStore):
            raise ManiculeError(
                "source lifecycle operations require the built-in workspace-scoped SQLite store"
            )
        return store

    async def export_corpus(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> tuple[int, int]:
        """Write retained bytes and metadata, and nothing derived from them."""
        from manicule.core.retrieval import Filter  # noqa: PLC0415

        store = await self._runtime.documents()
        blobs = await self._runtime.blobs()
        await asyncio.to_thread(
            _prepare_archive_dir, target, allow_insecure_target=allow_insecure_target
        )
        selector = Filter(workspace_ids=frozenset({self._runtime.workspace}))
        entries: list[ArchiveEntry] = []
        written = 0
        offset = 0
        while True:
            page = await store.list_documents(selector, limit=200, offset=offset)
            if not page:
                break
            offset += len(page)
            for document in page:
                if document.original_ref is None:
                    continue
                data = await blobs.get(document.original_ref)
                if data is None:
                    continue
                blob_path = target / "blobs" / document.original_ref
                await asyncio.to_thread(_write_private, blob_path, data)
                written += len(data)
                entries.append(
                    ArchiveEntry(
                        source=document.source,
                        source_id=document.source_id,
                        uri=document.uri,
                        title=document.title,
                        media_type=document.media_type,
                        version_token=document.version_token,
                        blob=f"blobs/{document.original_ref}",
                    )
                )
        manifest = ArchiveManifest(
            version=ARCHIVE_VERSION,
            workspace=self._runtime.workspace,
            documents=tuple(entries),
        )
        await asyncio.to_thread(
            _write_private,
            target / ARCHIVE_MANIFEST,
            manifest.model_dump_json(indent=2).encode("utf-8"),
        )
        return len(entries), written

    async def workspaces(self) -> Sequence[tuple[str, str, str]]:
        from sqlalchemy import select  # noqa: PLC0415

        from manicule.storage import models  # noqa: PLC0415
        from manicule.storage.engine import session_factory  # noqa: PLC0415

        await self._runtime.documents()
        sessions = session_factory(self._runtime.require_engine())
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(models.Workspace.id, models.Workspace.name, models.Workspace.mode)
                )
            ).all()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def _key_summary(row: object) -> ApiKeySummary:
    """One key's record, without anything that could be used as one."""
    expires = getattr(row, "expires_at", None)
    return ApiKeySummary(
        id=str(getattr(row, "id", "")),
        name=str(getattr(row, "name", "")),
        prefix=str(getattr(row, "key_prefix", "")),
        role=str(getattr(row, "role", "")),
        workspace=str(getattr(row, "workspace_id", "")),
        created_at=_isoformat(getattr(row, "created_at", None)),
        expires_at=_isoformat(expires) or None,
        revoked=getattr(row, "revoked_at", None) is not None,
    )


def _isoformat(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else ""


def _length(value: object) -> int:
    """How many entries a JSON column holds, tolerating one that holds something else.

    The column is typed ``JsonValue``, so a row written before this table was used — or by
    hand — can legitimately hold a scalar. ``len`` on that is a ``TypeError`` from inside a
    listing, which is a worse outcome than reporting zero for a row that recorded no chunks.
    """
    return len(cast("list[object]", value)) if isinstance(value, list) else 0


def _prepare_archive_dir(target: Path, *, allow_insecure_target: bool = False) -> None:
    """Create the archive directory and its blob shard. Blocking, so it runs off the loop.

    The mode is the same question ``backup`` answers, so it is answered by the same function
    rather than by a second one that would drift: an archive is every retained source byte
    this workspace holds, and this one is written *to be carried somewhere*.
    """
    from manicule.storage.engine import secure_output_dir  # noqa: PLC0415 - a storage extra

    secure_output_dir(target, operation="export", allow_insecure=allow_insecure_target)
    (target / "blobs").mkdir(mode=0o700, exist_ok=True)


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data``, readable by nobody but the owner, at no point wider.

    ``Path.write_bytes`` creates at the process ``umask``, which is commonly ``0644``. Inside
    an archive directory that is ``0700`` nobody else can reach the file — but an archive
    exists to be carried, and a file copied out of that directory takes its own mode with it,
    not the directory's. The same reasoning put the backup snapshot at ``0600``.

    The mode is passed to :func:`os.open` *and* applied to the descriptor, because the first
    is honored only when the call creates the file — the trap this whole family of bugs is
    made of — and a re-export over yesterday's archive would otherwise keep yesterday's mode.
    Doing it through the descriptor means there is no window in which the path exists at a
    wider mode, which matters when ``--allow-insecure-target`` made the directory reachable.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        if os.name == "posix":
            os.fchmod(handle.fileno(), 0o600)
        handle.write(data)


def _read_blob(path: Path) -> bytes | None:
    """One archived blob, or ``None`` when the archive does not contain it.

    ``None`` rather than an exception: an archive missing one document's bytes is a partial
    archive to report on, not a reason to abandon the other nine hundred.
    """
    if not path.is_file():
        return None
    return path.read_bytes()


def _open_archive(path: Path) -> tuple[Path, ArchiveManifest]:
    """Locate an archive's manifest — a directory, or the manifest itself — and read it."""
    manifest_path = path / ARCHIVE_MANIFEST if path.is_dir() else path
    return manifest_path, _read_archive(manifest_path)


def _read_archive(manifest_path: Path) -> ArchiveManifest:
    """Read an export manifest, refusing one this build cannot read.

    A newer archive is refused by version rather than parsed optimistically. Importing a
    layout this build does not understand would silently drop whatever it did not recognize,
    and a partial corpus that reports a clean import is the failure worth preventing.

    Raises:
        PolicyError: The file is not a manifest, or declares a version this build does not
            read.
    """
    if not manifest_path.is_file():
        msg = f"{manifest_path} is not an export manifest"
        raise PolicyError(msg)
    try:
        manifest = ArchiveManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        msg = f"{manifest_path} is not an export manifest this build can read: {exc}"
        raise PolicyError(msg) from exc
    if manifest.version != ARCHIVE_VERSION:
        msg = (
            f"{manifest_path} declares archive version {manifest.version}, and this build "
            f"reads version {ARCHIVE_VERSION}. Import it with the manicule that wrote it."
        )
        raise PolicyError(msg)
    return manifest


_STATE_NAMES: Mapping[HealthState, CheckState] = {
    HealthState.OK: "ok",
    HealthState.DEGRADED: "degraded",
    HealthState.FAILING: "failing",
}
"""A component's health, in the vocabulary the surfaces report.

Two names for three states rather than one shared enum, because the surfaces also report
``unknown`` — a check that could not run — and no component ever says that about itself.
"""


def _state_name(state: HealthState) -> CheckState:
    return _STATE_NAMES[state]


__all__ = [
    "ARCHIVE_MANIFEST",
    "ARCHIVE_VERSION",
    "ArchiveEntry",
    "ArchiveManifest",
    "AssemblyError",
    "Runtime",
]
