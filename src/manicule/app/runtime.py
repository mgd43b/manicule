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
from collections.abc import Mapping
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
from manicule.plugins.manifest import ComponentKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.app.ports import (
        Answering,
        Conversing,
        DocumentSurface,
        Ingesting,
        Keys,
        Maintenance,
        Organising,
        Retrieving,
        Telemetry,
    )
    from manicule.config.settings import Settings
    from manicule.core.fingerprints import GlossaryFingerprint
    from manicule.core.protocols import Connector, Parser, VectorStore
    from manicule.ingest.middleware import MiddlewareRunner
    from manicule.ingest.pipeline import BlobSink, IngestPipeline, RunReport
    from manicule.ingest.ports import IngestStore
    from manicule.ingest.reindex import GlossarySweep, ReindexReport, StaleSweep
    from manicule.plugins.registry import Discovery

ARCHIVE_MANIFEST = "manicule-export.json"
"""The file that makes an exported directory an archive rather than a pile of blobs."""

ARCHIVE_VERSION = 1
"""Bumped when the archive layout changes in a way an older import cannot read."""

KEY_PREFIX = "mnk_"
"""What every API key starts with.

A recognisable prefix so a leaked key is greppable — by its owner, and by a secret scanner
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

    def __init__(self, settings: Settings, *, discovery: Discovery | None = None) -> None:
        self._settings = settings
        self._container = build_container(settings, discovery=discovery)
        self._slots: dict[str, _Lazy] = {}
        self._engine: AsyncEngine | None = None
        self._migrated = False

    # --- lifecycle --------------------------------------------------------------------------

    @classmethod
    def open(cls, **overrides: Any) -> Runtime:  # noqa: ANN401 - mirrors Settings' own fields
        """Load configuration, verify it against what is installed, and return the runtime.

        Raises:
            ConfigError: The configuration is malformed.
            PolicyError: It is individually valid and jointly unrunnable, or it names a
                component nothing installed provides. Everything wrong is listed at once.
        """
        return cls(load_settings(**overrides))

    async def __aenter__(self) -> Self:
        return self

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
            await self._container.aclose()
        except Exception as during_teardown:
            if pending is None:
                raise
            pending.add_note(f"shutting down also failed: {during_teardown}")
        finally:
            if self._engine is not None:
                await self._engine.dispose()
                self._engine = None

    # --- what the service is given ----------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def workspace(self) -> str:
        return self._settings.workspace

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

    async def prepared_vectors(self) -> VectorStore:
        """The vector store, committed to the configured embedder's space.

        Both paths that touch vectors for real — ingest and retrieval — come through here, and
        this is the only place ``ensure_ready`` is called. Without it the store never learns
        which space it holds: the first upsert fails, the document is recorded ``failed`` at
        the ``store`` stage, and nothing about the message says the index was never prepared.

        It is also where a directory holding another model's vectors is refused, before a
        single vector is written into a space it does not belong to.
        """
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

    async def organisation(self) -> Organising:
        """Collections, tags and the trash.

        The **same object** the document store is. Two handles over one workspace would be two
        opinions about which documents are live, and the collection reads join against exactly
        the predicate the document reads use.
        """
        return await self._once("organisation", self._build_organisation)

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
        from manicule.storage.migrator import upgrade  # noqa: PLC0415

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
        if not self._migrated:
            # Before the first read, always. A query against an un-migrated database fails in
            # whatever way SQLite happens to fail, at whichever statement happens to run first.
            await upgrade(engine)
            self._migrated = True
        ensure = getattr(store, "ensure_workspace", None)
        if ensure is not None:
            await ensure()
        return store

    async def _build_vectors(self) -> VectorStore:
        return await self._container.aget(keys.VECTOR_STORE)

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
        await self.prepared_vectors()
        return await build_retriever(self._container)

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

    async def _build_organisation(self) -> Organising:
        from manicule.app.ports import Organising as Surface  # noqa: PLC0415

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
        return BlobStore(self.require_engine(), self._settings.data_dir)

    async def pipeline(self) -> IngestPipeline:
        """The ingest pipeline, refused before construction if it cannot write to this index."""
        return await self._once("pipeline", self._build_pipeline)

    async def _build_pipeline(self) -> IngestPipeline:
        from manicule.ingest.pipeline import IngestPipeline  # noqa: PLC0415
        from manicule.ingest.refusals import check_before_run  # noqa: PLC0415
        from manicule.ingest.workers import WorkerPool, worker_config  # noqa: PLC0415

        settings = self._settings
        store = await self.documents()
        chunker = await self._container.aget(keys.CHUNKER)
        embedder = await self._container.aget(keys.EMBEDDER)
        vectors = await self.prepared_vectors()
        # Through the accessor rather than assembled here, so that the pipeline's chain and the
        # one glossary repair compares against are one reading of one configuration. Two
        # readings that agreed today would part company the first time either grew a filter.
        middleware = await self.middleware()
        fingerprint = chunker.fingerprint.with_middleware(middleware.declarations())
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
        )
        pool = WorkerPool(
            worker_config(settings),
            workers=settings.ingest.parse_workers,
            timeout_s=settings.ingest.parse_timeout_s,
            poll_interval_s=settings.ingest.memory_poll_interval_s,
            max_documents=settings.ingest.max_documents_per_worker,
        )
        return IngestPipeline(
            store=store,  # pyright: ignore[reportArgumentType] - the store satisfies IngestStore
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
            max_fetch_bytes=settings.ingest.max_fetch_bytes,
            target_batch_tokens=settings.ingest.target_batch_tokens,
            max_embed_batch=settings.ingest.max_embed_batch,
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
        self, path: Path, *, name: str, limit: int | None = None, force: bool = False
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
            return await self._forced(connector, pipeline, limit=limit)
        return await pipeline.run(connector, limit=limit)

    async def _forced(
        self, connector: Connector, pipeline: IngestPipeline, *, limit: int | None
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
                if limit is not None and report.discovered >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - an enumeration failure is not a crash
            report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
        return report

    async def sync(self, connector: str, *, limit: int | None = None) -> RunReport:
        pipeline = await self._runtime.pipeline()
        return await pipeline.run(await self._runtime.connector(connector), limit=limit)

    async def connector(self, name: str) -> Connector:
        """The connector :meth:`sync` would run, handed back instead of run.

        One line, and it is deliberately the same line :meth:`sync` opens with rather than a
        parallel construction beside it. The container caches per instance, so a caller that
        asks for a source's profiles and then syncs it gets one object and cannot observe two
        readings of one configuration.
        """
        return await self._runtime.connector(name)

    async def reindex(self, document_id: str) -> ReindexReport:
        from manicule.ingest.reindex import reindex_document  # noqa: PLC0415

        store = await self._runtime.documents()
        return await reindex_document(
            document_id,
            store=store,  # pyright: ignore[reportArgumentType] - the store satisfies IngestStore
            pipeline=await self._runtime.pipeline(),
            blobs=await self._runtime.blobs(),
        )

    async def reparse_stale(self, *, batch: int, dry_run: bool = False) -> StaleSweep:
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

    async def redetect_stale_glossary(self, *, batch: int, dry_run: bool = False) -> GlossarySweep:
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
            outcomes = await pipeline.ingest_raw(
                raw,
                source=entry.source,
                version_token=entry.version_token,
                title=entry.title,
                force=force,
            )
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
        read into a write, serialising the whole API behind SQLite's single writer, and
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
                # that has no business travelling with it.
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

    async def reset_index(self) -> tuple[int, int, bool]:
        """Delete every document, chunk and vector this workspace owns.

        Vectors are removed **per document, here**, rather than left to the tombstone sweep.
        A reset that returned with the vectors still present would leave a search answering
        from an index the caller was told had been emptied — and the sweep runs on a cadence,
        so "eventually" could be an hour.
        """
        from manicule.core.retrieval import Filter  # noqa: PLC0415

        store = await self._runtime.documents()
        documents = await store.count_documents()
        chunks = await store.count_chunks()
        vectors = await self._runtime.vectors()
        removed = False
        selector = Filter(workspace_ids=frozenset({self._runtime.workspace}))
        while True:
            page = await store.list_documents(selector, limit=200)
            if not page:
                break
            for document in page:
                try:
                    await vectors.delete_document(document.id)
                except ManiculeError:
                    # An index that never received a vector has no table to delete from, and
                    # that is not a failed reset. Recorded rather than raised, so the caller
                    # is told what actually happened.
                    removed = removed or False
                else:
                    removed = True
                await store.delete_document(document.id)
        return documents, chunks, removed

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
    is honoured only when the call creates the file — the trap this whole family of bugs is
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
    layout this build does not understand would silently drop whatever it did not recognise,
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
