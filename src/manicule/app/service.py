"""The application service: every operation manicule offers, once.

Both surfaces are adapters over this class and neither contains any behaviour of its own.
That is not tidiness — it is the only way the two can be checked against each other. A rule
that lives in the CLI is a rule the MCP tool does not have, and the tool is the one an
assistant reaches for unattended.

Three properties hold for every method here:

**Every operation is workspace-scoped, and says so.** The filter carries the workspace, and
what comes back is checked again on the way out (:mod:`manicule.app.tenancy`) against an
identity that could only have been minted for this tenant.

**Nothing expensive is built until it is needed.** The backend's accessors are async and
lazy, so ``doctor`` does not load a model runtime and ``stats`` does not load a tokenizer.

**Results are payload models, never dictionaries.** The shape is defined in
:mod:`manicule.app.results` and dumped by one function, so ``--json`` and an MCP tool's
structured content are the same bytes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import time
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from manicule.app import results as r
from manicule.app.tenancy import CrossWorkspaceError, require_owned, require_owns
from manicule.config.loader import load_settings
from manicule.config.settings import (
    AuthMode,
    Role,
    Settings,
    config_file,
    looks_secret,
)
from manicule.core.content import DocumentStatus
from manicule.core.errors import ConfigError, ManiculeError, UnknownEntityError
from manicule.core.ids import document_id
from manicule.core.retrieval import Filter, Query, RetrievalProfile
from manicule.core.version import CORE_VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from manicule.app.ports import Backend, Conversing
    from manicule.core.content import Chunk, Document
    from manicule.core.organisation import Collection, Tag
    from manicule.generation.answers import AnswerEnvelope, AnswerEvent, Citation
    from manicule.generation.ports import ConversationRecord
    from manicule.ingest.pipeline import RunReport
    from manicule.plugins.registry import Discovery
    from manicule.retrieval.retriever import RetrievalResult

_log = logging.getLogger("manicule.app")
"""For the one thing here that is reported rather than raised: a telemetry write that failed.

A module logger rather than a print or a swallowed exception, so an operator who has configured
logging sees it and one who has not is not spammed by a library.
"""

DEFAULT_SOURCE = "local"
"""The source name ``index_path`` uses when none is given.

A constant rather than something derived from the directory, because a document's identity is
``(workspace, source, source_id)``: a source name that changed with the working directory
would re-index the same file as a new document every time it was indexed from somewhere else.
"""


@dataclass(slots=True)
class AskAside:
    """The two facts about one answer that live on neither the events nor the envelope.

    An async generator cannot also return a value, and hanging these off every event would
    make each consumer carry a record it mostly does not want. So the caller passes one in
    and reads it afterwards, which is the same shape
    :class:`~manicule.generation.answering.AnswerResult` uses one layer down and for the same
    reason.
    """

    confidence_band: str | None = None
    """The retrieval's confidence band. Not on the envelope, which carries only the score."""

    message_id: str | None = None
    """Where the answer was persisted, when there was a conversation to persist it to."""

    payload: r.AnswerResultPayload | None = None
    """The settled result, built by :meth:`ApplicationService.ask_stream` when the run ends.

    Here rather than assembled by each caller, because there are now two consumers of the
    stream — :meth:`ApplicationService.ask` and the HTTP surface's SSE and websocket paths —
    and a payload rebuilt at the surface is a second answer to "what did this run produce".
    It is filled after ``message_id`` is known, so a streamed answer carries the id it was
    persisted under exactly as the non-streaming form does.
    """


class ApplicationService:
    """Every operation, over a backend that supplies the parts."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> Backend:
        """What this service was given.

        Public because whoever assembled the backend is entitled to look at it — a suite
        driving a fake, a diagnostic reading the discovery. It is read-only: the service
        cannot be repointed at a different one, because a service whose tenant could change
        mid-life is one whose workspace checks describe the wrong workspace.
        """
        return self._backend

    @property
    def settings(self) -> Settings:
        return self._backend.settings

    @property
    def workspace(self) -> str:
        """The tenant every operation on this service runs in."""
        return self._backend.workspace

    # --- questions ------------------------------------------------------------------------

    async def ask(
        self,
        question: str,
        *,
        profile: str | None = None,
        limit: int | None = None,
        sources: Sequence[str] = (),
        conversation_id: str | None = None,
        on_event: Callable[[AnswerEvent], None] | None = None,
    ) -> r.AnswerResultPayload:
        """Answer one question over the corpus, and return the settled result.

        The streaming form is :meth:`ask_stream`; this consumes it. Both go through exactly
        the same path, so a caller that cannot stream is not on a different code path with
        different guarantees.

        ``on_event`` is called with each event as it arrives, which is how a terminal shows an
        answer being written. It is a *view* hook: it cannot change the answer, and the
        payload is identical whether or not one is passed. That is the difference between a
        surface that renders progress and a surface that has its own answer path.
        """
        aside = AskAside()
        async for event in self.ask_stream(
            question,
            profile=profile,
            limit=limit,
            sources=sources,
            conversation_id=conversation_id,
            aside=aside,
        ):
            if on_event is not None:
                on_event(event)
        if aside.payload is None:  # pragma: no cover - the answer path always ends with `final`
            msg = "the answer stream ended without a final event"
            raise ManiculeError(msg)
        return aside.payload

    async def ask_stream(
        self,
        question: str,
        *,
        profile: str | None = None,
        limit: int | None = None,
        sources: Sequence[str] = (),
        conversation_id: str | None = None,
        aside: AskAside | None = None,
    ) -> AsyncIterator[AnswerEvent]:
        """Stream one answer, token by token, ending with the ``final`` event.

        The tenancy check happens **before** the model is called. Refusing after the answer
        exists would already have sent another tenant's passages to a provider, which is the
        disclosure the check is here for.

        ``aside`` collects the two facts that are true of the run but are on neither the
        events nor the envelope — the retrieval's confidence *band* and the id the answer was
        persisted under. An async generator cannot also return a value, and putting them on
        every event would make each consumer carry a record it mostly does not want.
        """
        from manicule.generation.answering import (  # noqa: PLC0415 - keeps a cold start cold
            AnswerRequest,
            AnswerResult,
            answering,
        )

        started = time.monotonic()
        query = self._query(question, limit=limit or 8, profile=profile, sources=sources)
        retriever = await self._backend.retriever()
        retrieved = await retriever.retrieve(query)
        await self._require_scoped_context(retrieved)
        # After the tenancy check and before the model. Recording first would put another
        # tenant's chunk ids into this workspace's telemetry on the exact path the check
        # exists to stop.
        await self._record_query(query, retrieved, started=started)

        answerer = await self._backend.answerer()
        confidence = retrieved.confidence
        request = AnswerRequest(
            query=query,
            context=retrieved.context,
            conversation_id=conversation_id,
            confidence=confidence.score if confidence else None,
            corpus_consulted=retrieved.cites_the_corpus,
        )
        result = AnswerResult()
        record = aside if aside is not None else AskAside()
        if confidence is not None:
            record.confidence_band = confidence.band.value
        envelope: AnswerEnvelope | None = None
        try:
            async with answering(answerer, request, result) as events:
                async for event in events:
                    if event.envelope is not None:
                        envelope = event.envelope
                    yield event
        finally:
            # After the answering context has closed, so `message_id` is the id the turn was
            # actually persisted under. Building the payload at the `final` event instead
            # would report `message_id: null` on every streamed answer, which is precisely the
            # field a client needs in order to send feedback about it.
            record.message_id = result.message_id
            if envelope is not None:
                record.payload = self._answer_payload(
                    question, envelope, record, started, conversation_id
                )

    async def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
        profile: str | None = None,
        sources: Sequence[str] = (),
        media_types: Sequence[str] = (),
    ) -> r.SearchResult:
        """Rank passages without asking a model anything."""
        started = time.monotonic()
        query = self._query(
            query_text, limit=limit, profile=profile, sources=sources, media_types=media_types
        )
        retriever = await self._backend.retriever()
        retrieved = await retriever.retrieve(query)
        candidates = list(retrieved.candidates or retrieved.context.passages)[:limit]
        documents = await self._require_scoped_chunks(candidate.chunk for candidate in candidates)
        await self._record_query(query, retrieved, started=started)
        hits = tuple(
            r.SearchHit(
                document_id=candidate.chunk.document_id,
                chunk_id=candidate.chunk.id,
                uri=documents[candidate.chunk.document_id].uri,
                title=documents[candidate.chunk.document_id].title,
                heading_path=candidate.chunk.heading_path,
                kind=candidate.chunk.kind.value,
                anchor=_json_object(candidate.chunk.anchor.model_dump(mode="json")),
                score=candidate.score,
                scores=dict(candidate.scores),
                text=candidate.chunk.text,
                token_count=candidate.chunk.token_count,
            )
            for candidate in candidates
        )
        confidence = retrieved.confidence
        return r.SearchResult(
            query=query_text,
            profile=query.profile.value,
            count=len(hits),
            hits=hits,
            confidence=confidence.score if confidence else None,
            confidence_band=confidence.band.value if confidence else None,
            confidence_reason=confidence.reason if confidence else "",
            route=retrieved.trace.route.value,
            cached=retrieved.trace.cached,
            truncated=retrieved.context.truncated,
            elapsed_ms=_millis(started),
        )

    # --- ingest ---------------------------------------------------------------------------

    async def index_path(
        self,
        path: Path | str,
        *,
        source: str = DEFAULT_SOURCE,
        limit: int | None = None,
        force: bool = False,
    ) -> r.IngestReport:
        """Index a file or a directory.

        Args:
            path: What to index. A directory is walked recursively.
            source: The source name documents are recorded under. Part of their identity, so
                changing it re-indexes rather than updates.
            limit: Stop after this many discovered documents.
            force: Re-parse documents whose content has not changed.

        Raises:
            UnknownEntityError: The path does not exist.
        """
        started = time.monotonic()
        target = _local(path)
        if not await asyncio.to_thread(target.exists):
            msg = f"no such file or directory: {target}"
            raise UnknownEntityError(msg)
        ingestion = await self._backend.ingestion()
        report = await ingestion.index_path(
            await asyncio.to_thread(target.resolve), name=source, limit=limit, force=force
        )
        return _ingest_payload(report, started)

    async def index_changes(
        self,
        changed: Sequence[Path | str],
        *,
        source: str = DEFAULT_SOURCE,
        removed: Sequence[Path | str] = (),
    ) -> r.IngestReport:
        """Apply one batch of filesystem changes: index what appeared, drop what went.

        The service does this rather than the watch loop, because deleting a document means
        deriving its id from ``(workspace, source, source_id)`` — and an identity computed in a
        surface is an identity that can be computed differently in the next surface. A watcher
        hands over paths; what a path *is* stays here.
        """
        started = time.monotonic()
        totals = r.IngestReport(connector=source)
        counts: dict[str, int] = {}
        indexed = skipped = failed = 0
        for path in changed:
            report = await self.index_path(path, source=source)
            indexed += report.ingested
            skipped += report.skipped
            failed += report.failed
            for status, count in report.by_status.items():
                counts[status] = counts.get(status, 0) + count
        gone = 0
        for path in removed:
            identifier = document_id(self.workspace, source, str(_local(path).resolve()))
            try:
                await self.document_delete(identifier)
            except UnknownEntityError:
                # A path that was never indexed, or one already removed. A watcher reports
                # every deletion it sees, including of files manicule never held.
                continue
            gone += 1
        if gone:
            counts["deleted"] = gone
        return totals.model_copy(
            update={
                "discovered": len(list(changed)),
                "ingested": indexed,
                "skipped": skipped,
                "failed": failed,
                "by_status": counts,
                "elapsed_ms": _millis(started),
            }
        )

    async def connector_sync(self, name: str, *, limit: int | None = None) -> r.IngestReport:
        """Run one configured connector.

        Raises:
            UnknownEntityError: Configuration has no connector by that name.
        """
        started = time.monotonic()
        if name not in self.settings.connectors:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {name!r}. Configured: {known}"
            raise UnknownEntityError(msg)
        ingestion = await self._backend.ingestion()
        report = await ingestion.sync(name, limit=limit)
        return _ingest_payload(report, started)

    async def connector_login(
        self, name: str, *, cookies: str = "", forget: bool = False
    ) -> r.ConnectorSignedIn:
        """Capture the browser session a Confluence source authenticates with, or forget it.

        The session is proved against the instance before it is stored, because a cookie copied
        short, copied from the wrong tab or copied from a session that had already timed out is
        indistinguishable from a working one until something uses it — and otherwise the first
        thing to use it would be the first page of the next sync.

        manicule never sees the password. The caller supplies cookies from a browser that is
        already signed in; there is no parameter here that could carry a password and no code
        path that would accept one.

        Args:
            name: The configured source. Its type must be the Confluence connector, which is
                the only one that authenticates this way.
            cookies: The ``Cookie`` header from a signed-in browser.
            forget: Remove the stored session instead of capturing one.

        Raises:
            UnknownEntityError: Configuration has no connector by that name.
            ConfigError: That connector is not a Confluence source, or the paste carried no
                cookies, or the instance would not confirm who the session belongs to.
        """
        from manicule.connectors.config import (  # noqa: PLC0415 - no HTTP stack at import
            CONNECTOR_NAME,
            ConfluenceConfig,
        )
        from manicule.connectors.sessions import capture, default_store  # noqa: PLC0415

        # Settings directly rather than ``Runtime.connector(name)``, which every other
        # connector operation uses. That builds the connector, and building it resolves the
        # credential — which is the thing this operation exists to capture, and which by
        # definition is not there yet the first time somebody runs this.
        configured = self.settings.connectors.get(name)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {name!r}. Configured: {known}"
            raise UnknownEntityError(msg)
        if configured.type != CONNECTOR_NAME:
            msg = (
                f"{name!r} is a {configured.type!r} source, and a browser session is how the "
                f"{CONNECTOR_NAME!r} connector authenticates against an instance behind an "
                f"identity provider. Nothing else has a session to capture."
            )
            raise ConfigError(msg)
        config = ConfluenceConfig.model_validate(configured.options)
        store = default_store()
        if forget:
            store.forget(config.base_url)
            return r.ConnectorSignedIn(
                name=name,
                base_url=config.base_url,
                account="",
                captured_at="",
                expires_at="",
                stored_in=store.describe(),
                forgotten=True,
            )
        session = await capture(config, cookies, store=store)
        expires = session.captured_at + timedelta(hours=config.session_max_age_hours)
        return r.ConnectorSignedIn(
            name=name,
            base_url=config.base_url,
            account=session.account,
            captured_at=session.captured_at.isoformat(),
            expires_at=expires.isoformat(),
            stored_in=store.describe(),
        )

    async def connector_list(self) -> r.ConnectorList:
        """Every configured source, with what the last run recorded."""
        store = await self._backend.documents()
        discovery = self._backend.discovery
        installed: set[str] = (
            set(discovery.registry.names(_connector_kind())) if discovery else set()
        )
        summaries: list[r.ConnectorSummary] = []
        for name, configured in sorted(self.settings.connectors.items()):
            metadata = await store.connector_metadata(name)
            summaries.append(
                r.ConnectorSummary(
                    name=name,
                    type=configured.type,
                    enabled=configured.enabled,
                    schedule_s=configured.schedule_s,
                    installed=configured.type in installed if discovery else True,
                    last_synced_at=_text_or_none(metadata.get("last_synced_at")),
                    status=_text(metadata.get("status")),
                    documents=await store.count_documents(source=name),
                )
            )
        return r.ConnectorList(count=len(summaries), connectors=tuple(summaries))

    # --- documents ------------------------------------------------------------------------

    async def document_list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
        media_type: str | None = None,
    ) -> r.DocumentList:
        """A page of this workspace's documents, newest first."""
        store = await self._backend.documents()
        selector = Filter(
            workspace_ids=frozenset({self.workspace}),
            sources=frozenset({source}) if source else frozenset(),
            media_types=frozenset({media_type}) if media_type else frozenset(),
        )
        found = require_owned(
            self.workspace, await store.list_documents(selector, limit=limit, offset=offset)
        )
        return r.DocumentList(
            count=len(found),
            limit=limit,
            offset=offset,
            documents=tuple(_summary(document) for document in found),
        )

    async def document_get(self, document_id: str, *, chunks: bool = False) -> r.DocumentDetail:
        """One document, optionally with its stored chunks.

        Raises:
            UnknownEntityError: No live document with that id **in this workspace**. The
                message never says whether it exists elsewhere, because that answer is itself
                a cross-tenant disclosure.
        """
        store = await self._backend.documents()
        document = await store.get_document(document_id)
        if document is None:
            msg = (
                f"no live document {document_id!r} in workspace {self.workspace!r}. A "
                f"soft-deleted document has to be restored before it can be read."
            )
            raise UnknownEntityError(msg)
        require_owns(self.workspace, document)
        stored: Sequence[Chunk] = await store.document_chunks(document_id) if chunks else ()
        return r.DocumentDetail(
            document=_summary(document, chunk_count=len(stored) if chunks else None),
            chunks=tuple(
                r.DocumentChunk(
                    id=chunk.id,
                    position=chunk.position,
                    kind=chunk.kind.value,
                    heading_path=chunk.heading_path,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    anchor=_json_object(chunk.anchor.model_dump(mode="json")),
                )
                for chunk in stored
            ),
        )

    async def document_delete(self, document_id: str, *, hard: bool = False) -> r.DocumentDeleted:
        """Remove a document, into the trash by default.

        Raises:
            UnknownEntityError: No live document with that id in this workspace.
        """
        store = await self._backend.documents()
        document = await store.get_document(document_id)
        if document is None:
            msg = f"no live document {document_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)
        require_owns(self.workspace, document)
        if hard:
            await store.delete_document(document_id)
        else:
            await store.soft_delete_document(document_id)
        return r.DocumentDeleted(
            document_id=document_id, deleted=True, mode="hard" if hard else "soft"
        )

    async def document_reindex(self, document_id: str) -> r.DocumentReindexed:
        """Re-parse one document from its retained bytes. Touches no network."""
        store = await self._backend.documents()
        document = await store.get_document(document_id)
        if document is None:
            msg = f"no live document {document_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)
        require_owns(self.workspace, document)
        ingestion = await self._backend.ingestion()
        report = await ingestion.reindex(document_id)
        detail = "; ".join([*report.unrepairable, *report.failures])
        status = "reindexed" if report.documents else ("failed" if detail else "unchanged")
        return r.DocumentReindexed(
            document_id=document_id, status=status, chunks=report.chunks, detail=detail
        )

    # --- state ----------------------------------------------------------------------------

    async def index_status(self) -> r.IndexStatus:
        """What is in the index and what it was built with.

        Deliberately a report on what the store already knows. Trends, alerting and history
        belong to Operations; a surface that invented them here would be a second, weaker
        copy of that subsystem.
        """
        store = await self._backend.documents()
        maintenance = await self._backend.maintenance()
        statistics = await store.document_statistics()
        fingerprints = await store.index_fingerprints()
        return r.IndexStatus(
            documents=await store.count_documents(),
            chunks=await store.count_chunks(),
            by_status=dict(statistics.get("by_status", {})),
            embedding=fingerprints.embed.describe() if fingerprints.embed else "",
            chunking=fingerprints.chunk.describe() if fingerprints.chunk else "",
            # The canonical form as well as the readable one. They answer different questions:
            # one is for a person deciding whether this is the model they meant, the other is
            # the string a re-embed compares and must not be prettied.
            embed_fingerprint=fingerprints.embed.canonical() if fingerprints.embed else None,
            chunk_fingerprint=fingerprints.chunk.canonical() if fingerprints.chunk else None,
            schema_revision=await maintenance.schema_revision(),
            data_dir=str(self.settings.data_dir),
        )

    async def ready(self) -> bool:
        """Whether this installation can actually serve a question.

        Deliberately a **boolean**, and deliberately here rather than in the surface that asks.
        "Ready" is a decision — it means the store opens and answers — and a readiness probe
        that decided it for itself would be a second opinion about what a working installation
        is, on the one endpoint an orchestrator uses to restart things.

        It reports nothing about the corpus. A probe is reachable by whatever can reach the
        port, and "412 documents" is a corpus fingerprint.
        """
        try:
            store = await self._backend.documents()
            await store.count_documents()
        except (ManiculeError, ValueError, OSError):
            return False
        return True

    async def stats(self) -> r.Stats:
        """Counts, grouped the three ways anybody asks for."""
        store = await self._backend.documents()
        statistics = await store.document_statistics()
        return r.Stats(
            documents=await store.count_documents(),
            chunks=await store.count_chunks(),
            by_source=dict(statistics.get("by_source", {})),
            by_media_type=dict(statistics.get("by_media_type", {})),
            by_status=dict(statistics.get("by_status", {})),
        )

    async def doctor(self) -> r.Diagnosis:
        """Check what can be checked without building anything expensive.

        Every check answers one question and says what to do when the answer is bad. It does
        not load a model runtime, does not open a provider connection and does not read a
        document — a diagnostic that needs the system working is not a diagnostic.
        """
        checks: list[r.Check] = [
            self._configuration_check(),
            self._transport_check(),
            self._plugin_check(),
        ]
        checks.append(await self._storage_check())
        # After the storage check, which is what creates the data directory on a fresh
        # install. Asking about the modes of a directory that does not exist yet would report
        # "not created" on every first run, which reads as a fault and is not one.
        checks.append(await self._permissions_check())
        checks.append(await self._index_check())
        checks.extend(await self._backend.component_checks())
        worst: r.CheckState = "ok"
        for check in checks:
            if _severity(check.state) > _severity(worst):
                worst = check.state
        return r.Diagnosis(state=worst, checks=tuple(checks))

    def _configuration_check(self) -> r.Check:
        problems = self.settings.policy_problems()
        if not problems:
            return r.Check(name="configuration", state="ok", detail="no policy conflicts")
        return r.Check(name="configuration", state="failing", detail="; ".join(problems))

    def _transport_check(self) -> r.Check:
        transport = self.settings.security.transport
        if transport.is_loopback:
            return r.Check(
                name="transport",
                state="ok",
                detail=f"bound to {transport.bind_host}, reachable only from this machine",
            )
        if self.settings.security.auth.mode is AuthMode.NONE:
            return r.Check(
                name="transport",
                state="failing",
                detail=(
                    f"security.transport.bind_host is {transport.bind_host!r} with no "
                    f"authentication. Bind 127.0.0.1, or set security.auth.mode."
                ),
            )
        return r.Check(
            name="transport",
            state="degraded",
            detail=(
                f"bound to {transport.bind_host}, which is reachable from the network. "
                f"Authentication is on ({self.settings.security.auth.mode.value})."
            ),
        )

    def _plugin_check(self) -> r.Check:
        discovery = self._backend.discovery
        if discovery is None:
            return r.Check(name="plugins", state="unknown", detail="discovery has not run")
        disabled = f", {len(discovery.disabled)} disabled" if discovery.disabled else ""
        return r.Check(
            name="plugins",
            state="ok",
            detail=(
                f"{len(discovery.manifests)} plugin(s), "
                f"{len(discovery.registry)} component(s){disabled}"
            ),
        )

    async def _storage_check(self) -> r.Check:
        try:
            maintenance = await self._backend.maintenance()
            revision = await maintenance.schema_revision()
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(name="storage", state="failing", detail=f"{type(exc).__name__}: {exc}")
        if revision is None:
            return r.Check(
                name="storage",
                state="degraded",
                detail="the database carries no schema revision; run `manicule init`",
            )
        return r.Check(name="storage", state="ok", detail=f"schema at {revision}")

    async def _permissions_check(self) -> r.Check:
        """Whether anybody but the owner can read the data directory.

        **Failing, not degraded, and the distinction is the whole point.** With retained
        source bytes the data directory is a verbatim copy of the corpus (``docs/storage.md``
        §7.1), so a group- or world-readable one has already published every document indexed
        into it to everyone with an account on the machine. There is no reading of that which
        is a warning: nothing downstream degrades, and the exposure does not announce itself.

        The offending paths are named, because "your permissions are wrong" sends an operator
        to ``chmod -R``, and a data directory is the last place to run a recursive mode change
        from memory.
        """
        from manicule.storage.engine import exposure  # noqa: PLC0415 - a storage extra

        data_dir = self.settings.data_dir
        if os.name != "posix":
            return r.Check(
                name="permissions",
                state="unknown",
                detail=f"{data_dir} is on a platform with no POSIX file modes to check",
            )
        try:
            exposed = await asyncio.to_thread(exposure, data_dir)
        except OSError as exc:
            # "Cannot be examined" is not "is exposed", and reporting it as the latter would
            # send an operator to chmod a path that may not be there.
            return r.Check(
                name="permissions",
                state="unknown",
                detail=f"{data_dir} could not be examined: {exc}",
            )
        if not exposed:
            return r.Check(
                name="permissions",
                state="ok",
                detail=f"{data_dir} is readable only by the account running manicule",
            )
        return r.Check(
            name="permissions",
            state="failing",
            detail=(
                f"{data_dir} carries group or other permissions ({exposed:03o}), so it is "
                f"reachable by accounts other than the one running manicule. It holds the "
                f"retained source bytes of every indexed document, which makes this an "
                f"exposure of the corpus rather than a tidiness problem. Run "
                f"`chmod 0700 {data_dir}`."
            ),
        )

    async def _index_check(self) -> r.Check:
        try:
            store = await self._backend.documents()
            fingerprints = await store.index_fingerprints()
            documents = await store.count_documents()
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(name="index", state="failing", detail=f"{type(exc).__name__}: {exc}")
        if fingerprints.is_empty:
            return r.Check(
                name="index",
                state="ok" if documents == 0 else "degraded",
                detail=(
                    "empty index, ready for a first ingest"
                    if documents == 0
                    else f"{documents} document(s) but no recorded fingerprints"
                ),
            )
        return r.Check(
            name="index",
            state="ok",
            detail=f"{documents} document(s) in {fingerprints.vector_table or 'no vector table'}",
        )

    # --- configuration --------------------------------------------------------------------

    async def config_get(self, key: str = "") -> r.ConfigValue:
        """Read the effective configuration, redacted.

        Redacted always, and not as a courtesy: this is what an MCP tool returns to an
        assistant, so returning live secrets would hand out every API key and signing key to
        anything holding a tool call.

        Raises:
            UnknownEntityError: No such setting.
        """
        redacted = self.settings.redacted()
        if not key:
            return r.ConfigValue(key="", value=redacted, source=str(config_file()))
        value: JsonValue = redacted
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                msg = f"no such setting: {key!r}"
                raise UnknownEntityError(msg)
            value = value[part]
        return r.ConfigValue(key=key, value=value, source=str(config_file()))

    async def config_set(self, key: str, value: str) -> r.ConfigChange:
        """Write one setting to the config file, having validated the whole tree with it.

        ``value`` is parsed as JSON first and kept as a string when that fails, so
        ``rag.cache.enabled false`` sets a boolean and ``llm.model qwen2.5:14b`` sets a
        string without anybody quoting anything.

        A secret is refused rather than written. Credentials belong in the environment, where
        they are not copied into backups, exports or version control by accident.

        Raises:
            ConfigError: The key names no setting, the value does not validate, or the key is
                a credential.
        """
        if not key:
            msg = "config set needs a dotted key, for example 'rag.profile'"
            raise ConfigError(msg)
        parts = key.split(".")
        if looks_secret(parts[-1]):
            msg = (
                f"{key!r} is a credential. Set it in the environment instead — the config "
                f"file is copied into backups and exports, and a secret written here goes "
                f"with them."
            )
            raise ConfigError(msg)
        parsed = _parse_value(value)
        previous = await self._current_value(parts)

        path = config_file()
        # The whole read-modify-validate-write runs in a worker thread. It is blocking file
        # I/O in an async method, and pydantic-settings re-reads the file and the environment
        # while validating, so this is more than the one write it looks like.
        await asyncio.to_thread(
            _update_config, path, lambda document: _assign(document, parts, parsed)
        )
        return r.ConfigChange(key=key, previous=previous, value=parsed, path=str(path))

    async def _current_value(self, parts: Sequence[str]) -> JsonValue:
        value: JsonValue = self.settings.redacted()
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    # --- workspaces -----------------------------------------------------------------------

    async def workspace_list(self) -> r.WorkspaceList:
        """Every workspace this installation knows about.

        Counts are reported for the active workspace only. This handle is scoped to one
        tenant, and counting another's documents through it would be the breach the scope
        exists to prevent — reported as a number rather than as text, but a read all the same.
        """
        maintenance = await self._backend.maintenance()
        store = await self._backend.documents()
        rows = await maintenance.workspaces()
        known = {row[0] for row in rows}
        if self.workspace not in known:
            rows = [*rows, (self.workspace, self.workspace, self.settings.mode.value)]
        summaries: list[r.WorkspaceSummary] = []
        for identifier, name, mode in sorted(rows):
            active = identifier == self.workspace
            summaries.append(
                r.WorkspaceSummary(
                    id=identifier,
                    name=name,
                    mode=mode,
                    active=active,
                    documents=await store.count_documents() if active else None,
                )
            )
        return r.WorkspaceList(
            active=self.workspace, count=len(summaries), workspaces=tuple(summaries)
        )

    async def workspace_switch(self, name: str, *, create: bool = False) -> r.WorkspaceSwitched:
        """Record a different active workspace in the config file.

        It takes effect on the next start, and deliberately not on this one: half a process
        holding handles scoped to the old workspace and half to the new one is exactly the
        state in which a cross-tenant read stops being impossible.

        Raises:
            ConfigError: The name is empty.
            UnknownEntityError: No such workspace, and ``create`` was not asked for.
        """
        wanted = name.strip()
        if not wanted:
            msg = "a workspace name cannot be empty"
            raise ConfigError(msg)
        maintenance = await self._backend.maintenance()
        known = {row[0] for row in await maintenance.workspaces()}
        if wanted not in known and not create:
            listed = ", ".join(sorted(known)) or "none"
            msg = f"no workspace named {wanted!r}. Known: {listed}. Pass create to make it."
            raise UnknownEntityError(msg)
        previous = self.workspace
        change = await self.config_set("workspace", wanted)
        return r.WorkspaceSwitched(previous=previous, active=wanted, path=change.path)

    # --- plugins --------------------------------------------------------------------------

    async def plugin_list(
        self, *, registry: bool = False, query: str | None = None
    ) -> r.PluginList:
        """What is installed, and — when configuration allows it — what is offered.

        ``query`` filters the **offered** list by name or summary, case-insensitively. It is
        here rather than in a surface because a filter written twice is a filter that answers
        differently on two surfaces, and this one decides what an operator is shown before
        they choose something to install.
        """
        discovery = self._backend.discovery
        plugins: tuple[r.PluginSummary, ...] = ()
        disabled: tuple[str, ...] = ()
        if discovery is not None:
            by_plugin: dict[str, list[r.ComponentSummary]] = {}
            for record in discovery.registry.records():
                by_plugin.setdefault(record.plugin, []).append(
                    r.ComponentSummary(
                        kind=record.kind.value,
                        name=record.name,
                        plugin=record.plugin,
                        summary=record.summary,
                    )
                )
            plugins = tuple(
                r.PluginSummary(
                    name=manifest.name,
                    version=manifest.version,
                    core_version=manifest.core_version,
                    summary=manifest.summary,
                    enabled=True,
                    components=tuple(by_plugin.get(manifest.name, ())),
                )
                for manifest in discovery.manifests
            )
            disabled = discovery.disabled

        available: tuple[r.AvailablePlugin, ...] = ()
        error = ""
        settings = self.settings
        if registry:
            if not settings.plugins.allow_install:
                error = (
                    "plugins.allow_install is false, so the community registry is not "
                    "consulted. A plugin runs with this process's full authority, so "
                    "browsing a list of them is opt-in."
                )
            else:
                installed = {plugin.name for plugin in plugins}
                available, error = await _fetch_registry(settings.plugins.registry_url, installed)
                if query:
                    needle = query.casefold()
                    available = tuple(
                        entry
                        for entry in available
                        if needle in entry.name.casefold() or needle in entry.summary.casefold()
                    )
        return r.PluginList(
            count=len(plugins),
            plugins=plugins,
            disabled=disabled,
            available=available,
            registry_url=settings.plugins.registry_url,
            registry_error=error,
        )

    async def plugin_add(self, name: str) -> r.PluginChanged:
        """Enable an installed plugin.

        **manicule never installs one.** Installing a plugin runs its code with this
        process's full authority (``CONTRIBUTING.md``), and a surface an assistant can call
        unattended must not be able to fetch and execute a package. A plugin that is not
        installed is reported with the command that would install it, and nothing is run.

        Raises:
            UnknownEntityError: Nothing installed provides it, and the registry does not list
                it either.
        """
        discovery = self._backend.discovery
        installed = _installed_plugins(discovery)
        if name not in installed:
            available, _ = (
                await _fetch_registry(self.settings.plugins.registry_url, installed)
                if self.settings.plugins.allow_install
                else ((), "")
            )
            offered = next((entry for entry in available if entry.name == name), None)
            if offered is None:
                known = ", ".join(sorted(installed)) or "none"
                msg = (
                    f"no plugin named {name!r} is installed. Installed: {known}. Install the "
                    f"distribution that provides it and run this again."
                )
                raise UnknownEntityError(msg)
            return r.PluginChanged(
                name=name,
                enabled=False,
                installed=False,
                detail=(
                    f"{name} is offered by the registry but is not installed. Install it with "
                    f"`uv tool install --with {offered.package or name} manicule`, then run "
                    f"this again. manicule does not run a package manager for you: a plugin "
                    f"runs with this process's full authority."
                ),
            )
        change = await self._set_plugin_lists(name, enabled=True)
        return r.PluginChanged(
            name=name, enabled=True, installed=True, path=change, detail="enabled at next start"
        )

    async def plugin_remove(self, name: str) -> r.PluginChanged:
        """Disable a plugin. The distribution stays installed and is not touched.

        Raises:
            UnknownEntityError: Nothing installed provides it.
        """
        discovery = self._backend.discovery
        installed = _installed_plugins(discovery)
        if discovery is not None and name not in installed:
            known = ", ".join(sorted(installed)) or "none"
            msg = f"no plugin named {name!r} is installed. Installed: {known}"
            raise UnknownEntityError(msg)
        change = await self._set_plugin_lists(name, enabled=False)
        return r.PluginChanged(
            name=name,
            enabled=False,
            installed=True,
            path=change,
            detail="disabled at next start; the distribution is untouched",
        )

    async def _set_plugin_lists(self, name: str, *, enabled: bool) -> str:
        path = config_file()
        await asyncio.to_thread(_update_config, path, _plugin_mutation(name, enabled=enabled))
        return str(path)

    # --- api keys -------------------------------------------------------------------------

    async def api_key_create(
        self, name: str, *, role: str = "member", expires_days: int | None = None
    ) -> r.ApiKeyIssued:
        """Mint an API key for this workspace.

        Raises:
            ConfigError: The name is empty, or the role is not one manicule has.
        """
        label = name.strip()
        if not label:
            msg = "an API key needs a name, so that it can be revoked without guessing"
            raise ConfigError(msg)
        try:
            chosen = Role(role)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in Role)
            msg = f"no such role {role!r}. Available: {allowed}"
            raise ConfigError(msg) from exc
        keys = await self._backend.keys()
        summary, secret = await keys.issue(label, role=chosen.value, expires_days=expires_days)
        # The record, never the secret. An audit trail that quoted the credential it was
        # recording the creation of would be a copy of every key ever minted.
        await self._audit(
            "api_key.created",
            details={"id": summary.id, "name": summary.name, "role": summary.role},
        )
        return r.ApiKeyIssued(key=summary, secret=secret)

    async def api_key_list(self) -> r.ApiKeyList:
        """Every key in this workspace. Never a secret — only digests are stored."""
        keys = await self._backend.keys()
        listed = tuple(await keys.list_keys())
        return r.ApiKeyList(count=len(listed), keys=listed)

    async def api_key_revoke(self, name_or_id: str) -> r.ApiKeyRevoked:
        """Revoke a key.

        Raises:
            UnknownEntityError: No key by that name or id **in this workspace**.
        """
        keys = await self._backend.keys()
        summary = await keys.revoke(name_or_id)
        await self._audit("api_key.revoked", details={"id": summary.id, "name": summary.name})
        return r.ApiKeyRevoked(id=summary.id, name=summary.name, revoked=True)

    # --- operations the command line owns -------------------------------------------------

    async def backup(
        self, target: Path | str, *, allow_insecure_target: bool = False
    ) -> r.BackupReport:
        """Take a consistent copy of the data directory.

        Refuses a group- or world-readable target unless ``allow_insecure_target`` says
        otherwise, for the reason ``docs/storage.md`` §7.1 gives and ``doctor``'s permissions
        check already acts on: a snapshot is a verbatim copy of every indexed document, and
        backup is the routine operation, so an unprotected one is the likeliest way that copy
        ends up somewhere it should not be.
        """
        maintenance = await self._backend.maintenance()
        manifest = await maintenance.backup(
            _local(target), allow_insecure_target=allow_insecure_target
        )
        files = _as_list(manifest.get("files"))
        return r.BackupReport(
            path=str(_local(target)),
            created_at=_text(manifest.get("created_at")),
            files=len(files),
            bytes=sum(_int(entry.get("size")) for entry in files if isinstance(entry, dict)),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            # `alembic_revision` is what the manifest calls it. Read by the key the writer
            # uses rather than by the name this payload gives it, because a key that is
            # merely plausible reads as `null` and nothing says the lookup missed.
            schema_revision=_text_or_none(manifest.get("alembic_revision")),
            counts={
                str(key): _int(count) for key, count in _as_mapping(manifest.get("counts")).items()
            },
        )

    async def restore(self, source: Path | str, *, force: bool = False) -> r.RestoreReport:
        """Put a backup back over the data directory."""
        maintenance = await self._backend.maintenance()
        manifest = await maintenance.restore(_local(source), force=force)
        return r.RestoreReport(
            path=str(_local(source)),
            data_dir=str(self.settings.data_dir),
            files=len(_as_list(manifest.get("files"))),
        )

    async def export_corpus(self, target: Path | str) -> r.ExportReport:
        """Write a portable archive: retained bytes and metadata, never chunks or vectors.

        An archive that carried chunks would carry an index built by another chunker and
        another embedder, and dropping it into a store whose fingerprints say otherwise is
        the silent-mismatch failure the fingerprints exist to prevent. The importing
        installation re-derives both.
        """
        maintenance = await self._backend.maintenance()
        documents, chunks = await maintenance.export_corpus(_local(target))
        return r.ExportReport(path=str(_local(target)), documents=documents, chunks=chunks)

    async def import_corpus(self, source: Path | str, *, force: bool = False) -> r.IngestReport:
        """Ingest an exported archive, re-deriving chunks and vectors here."""
        started = time.monotonic()
        path = _local(source)
        if not await asyncio.to_thread(path.exists):
            msg = f"no such archive: {path}"
            raise UnknownEntityError(msg)
        ingestion = await self._backend.ingestion()
        return _ingest_payload(await ingestion.import_archive(path, force=force), started)

    async def reset_index(self) -> r.ResetReport:
        """Delete every document, chunk and vector in this workspace.

        Irreversible, and the surfaces make the caller say so: the CLI requires ``--yes`` and
        there is no MCP tool for it at all.
        """
        maintenance = await self._backend.maintenance()
        documents, chunks, vectors = await maintenance.reset_index()
        return r.ResetReport(documents=documents, chunks=chunks, vectors_removed=vectors)

    async def initialise(self, *, force: bool = False) -> r.InitReport:
        """Write a starting configuration, choosing what this machine can actually run.

        The hardware probe picks the embedding backend rather than recommending one, because
        a default that is wrong for the machine is discovered at the first ingest, and by then
        a corpus has been indexed with it.
        """
        path = config_file()
        if await asyncio.to_thread(path.exists) and not force:
            msg = f"{path} already exists. Pass force to overwrite it."
            raise ConfigError(msg)
        probe = hardware()
        settings = self.settings
        provider = "mlx" if probe["apple_silicon"] else "onnx"
        await asyncio.to_thread(_update_config, path, _initial_configuration(settings, provider))
        notes = [
            f"embedding backend {provider!r} chosen for {probe['machine']} on {probe['system']}",
            "bind host written as 127.0.0.1; a wider bind needs auth and an explicit flag",
        ]
        return r.InitReport(
            path=str(path),
            data_dir=str(settings.data_dir),
            embedding_provider=provider,
            embedding_model=settings.embedding.model,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            hardware=probe,
            notes=tuple(notes),
        )

    async def upgrade(
        self, *, version: str | None = None, skip_backup: bool = False
    ) -> r.UpgradeReport:
        """Take a backup and report exactly how to upgrade. It does not run a package manager.

        Upgrading means fetching and executing code. A surface that did that on request would
        be a remote-code-execution path reachable from a command that reads like a version
        bump, and a half-finished install of the thing holding the index is the worst possible
        moment to discover it. So the backup — the part that is dangerous to skip — happens
        here, and the install is a command a person runs.
        """
        target = version or "latest"
        backup_path: str | None = None
        if not skip_backup:
            destination = self.settings.data_dir / "backups" / f"pre-upgrade-{int(time.time())}"
            backup_path = (await self.backup(destination)).path
        specifier = f"manicule=={version}" if version else "manicule"
        return r.UpgradeReport(
            current=CORE_VERSION,
            target=target,
            latest=None,
            backup=backup_path,
            performed=False,
            detail=(
                f"run `uv tool install --force {specifier}` to upgrade. manicule does not "
                f"install itself: that is fetching and running code, and a failure part-way "
                f"leaves the installation holding your index broken."
            ),
        )

    # --- conversations --------------------------------------------------------------------

    async def conversation_create(self, *, title: str | None = None) -> r.ConversationSummary:
        """Start a conversation in this workspace."""
        store = await self._backend.conversations()
        identifier = await store.create_conversation(title=title)
        record = await store.get_conversation(identifier)
        if record is None:  # pragma: no cover - the row was just written
            msg = f"conversation {identifier!r} vanished between creation and reading"
            raise ManiculeError(msg)
        return _conversation(record)

    async def conversation_list(self, *, limit: int = 50, offset: int = 0) -> r.ConversationList:
        """A page of this workspace's live conversations, most recently touched first."""
        store = await self._backend.conversations()
        found = await store.list_conversations(limit=limit, offset=offset)
        return r.ConversationList(
            count=len(found),
            limit=limit,
            offset=offset,
            conversations=tuple(_conversation(record) for record in found),
        )

    async def conversation_messages(
        self, conversation_id: str, *, limit: int = 50
    ) -> r.ConversationMessages:
        """A conversation's turns, oldest first, as its **owner** reads them.

        Full citations, passage text included. The anonymous projection is a different method
        over a different query, and the two never meet: this one is reached only through a
        workspace-scoped handle.

        Raises:
            UnknownEntityError: No live conversation of that id in this workspace.
        """
        store = await self._backend.conversations()
        await self._require_conversation(store, conversation_id)
        turns = await store.history(conversation_id, limit=limit)
        return r.ConversationMessages(
            conversation_id=conversation_id,
            count=len(turns),
            turns=tuple(
                r.ConversationTurn(
                    role=turn.role,
                    content=turn.content,
                    citations=tuple(_citation(citation) for citation in turn.citations),
                )
                for turn in turns
            ),
        )

    async def conversation_rename(self, conversation_id: str, title: str) -> r.ConversationRenamed:
        """Retitle a conversation.

        Raises:
            ConfigError: The title is empty.
            UnknownEntityError: No live conversation of that id in this workspace.
        """
        wanted = title.strip()
        if not wanted:
            msg = "a conversation title cannot be empty"
            raise ConfigError(msg)
        store = await self._backend.conversations()
        if not await store.rename_conversation(conversation_id, wanted):
            raise _no_conversation(conversation_id, self.workspace)
        return r.ConversationRenamed(conversation_id=conversation_id, title=wanted)

    async def conversation_delete(self, conversation_id: str) -> r.ConversationDeleted:
        """Soft-delete a conversation, which also revokes any share link over it.

        Raises:
            UnknownEntityError: No live conversation of that id in this workspace.
        """
        store = await self._backend.conversations()
        if not await store.soft_delete_conversation(conversation_id):
            raise _no_conversation(conversation_id, self.workspace)
        await self._audit(
            "conversation.deleted",
            details={"conversation_id": conversation_id},
        )
        return r.ConversationDeleted(conversation_id=conversation_id, deleted=True)

    async def conversation_share(
        self, conversation_id: str, *, ttl_s: int | None = None
    ) -> r.ShareCreated:
        """Mint a share link, and return the only copy of its token.

        Every bound comes from :mod:`manicule.generation.sharing` and the store, not from
        here: sharing must be switched on, the requested lifetime is clamped to
        ``security.sharing.link_ttl_s``, and the store re-checks the ceiling because a
        :class:`~manicule.generation.sharing.ShareLink` is an ordinary value object a caller
        could have built by hand.

        Minting **replaces** any previous link, so the old token stops working immediately and
        the new one is a fresh snapshot.

        Raises:
            PolicyError: ``security.sharing.enabled`` is false.
            ValueError: The requested lifetime is not positive.
            UnknownEntityError: No live conversation of that id in this workspace.
        """
        from manicule.generation.sharing import (  # noqa: PLC0415 - only sharing needs it
            new_share,
            require_sharing_enabled,
        )

        sharing = self.settings.security.sharing
        require_sharing_enabled(sharing.enabled)
        link = new_share(
            conversation_id,
            ttl_s=ttl_s if ttl_s is not None else sharing.link_ttl_s,
            # Always passed. It was optional once, no production caller supplied it, and the
            # clamp therefore never ran — so a hundred-year link minted cleanly.
            maximum_ttl_s=sharing.link_ttl_s,
        )
        store = await self._backend.conversations()
        if not await store.create_share(link, maximum_ttl_s=sharing.link_ttl_s):
            raise _no_conversation(conversation_id, self.workspace)
        await self._audit(
            "conversation.shared",
            details={"conversation_id": conversation_id, "expires_at": link.expires_at.isoformat()},
        )
        return r.ShareCreated(
            conversation_id=conversation_id,
            token=link.token,
            path=link.path,
            expires_at=link.expires_at.isoformat(),
            shared_at=link.shared_at.isoformat(),
        )

    async def conversation_unshare(self, conversation_id: str) -> r.ShareRevoked:
        """Revoke a share link by clearing the stored hash.

        Raises:
            UnknownEntityError: No conversation of that id in this workspace.
        """
        store = await self._backend.conversations()
        if not await store.revoke_share(conversation_id):
            raise _no_conversation(conversation_id, self.workspace)
        await self._audit("conversation.unshared", details={"conversation_id": conversation_id})
        return r.ShareRevoked(conversation_id=conversation_id, revoked=True)

    async def shared_conversation(self, token: str) -> r.SharedConversation:
        """Read a shared conversation from its token, as an anonymous viewer.

        **The only path to conversation data that does not require a workspace membership**,
        and it is deliberately shaped so that holding the token is the whole of what it can
        do. The token is hashed before it reaches storage, the store resolves it in one
        statement with expiry, revocation, soft-delete and the snapshot boundary as predicates
        of that same statement, and what comes back is already citation *labels*.

        An unknown token, an expired one, a revoked one, a deleted conversation and sharing
        being switched off all produce the same empty result. Distinguishing them for an
        unauthenticated caller tells them which of their guesses was closest.
        """
        from datetime import UTC, datetime  # noqa: PLC0415 - only this method needs the clock

        from manicule.generation.sharing import hash_token  # noqa: PLC0415

        store = await self._backend.conversations()
        turns = await store.shared_conversation(
            hash_token(token) if token else "",
            now=datetime.now(UTC),
            # Read from configuration on the *read* path, not only at minting: an operator
            # switching sharing off has decided the disclosure already made is the problem.
            sharing_enabled=self.settings.security.sharing.enabled,
        )
        return r.SharedConversation(
            count=len(turns),
            turns=tuple(
                r.SharedTurnPayload(
                    role=turn.role,
                    content=turn.content,
                    citations=tuple(
                        r.SharedCitationLabel(
                            slot=label.slot,
                            title=label.title,
                            heading_path=label.heading_path,
                            location=label.location,
                            verification=label.verification.value,
                        )
                        for label in turn.citations
                    ),
                )
                for turn in turns
            ),
        )

    async def chat_feedback(
        self, message_id: str, *, feedback: str, reason: str | None = None, comment: str = ""
    ) -> r.FeedbackRecorded:
        """Rate one answer.

        Raises:
            ConfigError: The rating or the reason is not one manicule has.
            UnknownEntityError: No such message in this workspace.
        """
        from manicule.generation.ports import Feedback, FeedbackReason  # noqa: PLC0415

        try:
            rating = Feedback(feedback)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in Feedback)
            msg = f"no such feedback {feedback!r}. Available: {allowed}"
            raise ConfigError(msg) from exc
        chosen: FeedbackReason | None = None
        if reason:
            try:
                chosen = FeedbackReason(reason)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in FeedbackReason)
                msg = f"no such feedback reason {reason!r}. Available: {allowed}"
                raise ConfigError(msg) from exc
        store = await self._backend.conversations()
        if not await store.record_feedback(
            message_id, feedback=rating, reason=chosen, comment=comment
        ):
            msg = (
                f"no message {message_id!r} in workspace {self.workspace!r}. Feedback on an id "
                f"that matched nothing is accepted and never seen again, so it is refused."
            )
            raise UnknownEntityError(msg)
        return r.FeedbackRecorded(message_id=message_id, recorded=True, feedback=rating.value)

    async def _require_conversation(self, store: Conversing, conversation_id: str) -> None:
        if await store.get_conversation(conversation_id) is None:
            raise _no_conversation(conversation_id, self.workspace)

    # --- collections ----------------------------------------------------------------------

    async def collection_create(
        self, name: str, *, description: str | None = None
    ) -> r.CollectionSummary:
        """Create a collection. A duplicate name is refused rather than merged.

        Raises:
            ValueError: The name is empty once normalised.
            NameInUseError: A collection of that name already exists here.
        """
        store = await self._backend.organisation()
        return _collection(await store.create_collection(name, description=description))

    async def collection_list(self) -> r.CollectionList:
        """Every collection in this workspace."""
        store = await self._backend.organisation()
        found = await store.list_collections()
        return r.CollectionList(
            count=len(found), collections=tuple(_collection(item) for item in found)
        )

    async def collection_delete(self, collection_id: str) -> r.CollectionDeleted:
        """Delete a collection. The documents in it are untouched.

        Raises:
            UnknownEntityError: No such collection in this workspace.
        """
        store = await self._backend.organisation()
        await store.delete_collection(collection_id)
        return r.CollectionDeleted(collection_id=collection_id, deleted=True)

    async def collection_add(
        self, collection_id: str, document_ids: Sequence[str]
    ) -> r.CollectionMembership:
        """Add documents to a collection.

        Raises:
            UnknownEntityError: No such collection, or a document this workspace cannot see.
        """
        store = await self._backend.organisation()
        changed = await store.add_to_collection(collection_id, list(document_ids))
        return r.CollectionMembership(
            collection_id=collection_id, changed=changed, document_ids=tuple(document_ids)
        )

    async def collection_remove(
        self, collection_id: str, document_ids: Sequence[str]
    ) -> r.CollectionMembership:
        """Remove documents from a collection.

        Raises:
            UnknownEntityError: No such collection in this workspace.
        """
        store = await self._backend.organisation()
        changed = await store.remove_from_collection(collection_id, list(document_ids))
        return r.CollectionMembership(
            collection_id=collection_id, changed=changed, document_ids=tuple(document_ids)
        )

    async def collection_documents(
        self, collection_id: str, *, limit: int = 50, offset: int = 0
    ) -> r.DocumentList:
        """A page of a collection's documents, checked on the way out like every other read.

        Raises:
            UnknownEntityError: No such collection in this workspace.
            CrossWorkspaceError: A document came back whose id was not minted here.
        """
        store = await self._backend.organisation()
        if await store.get_collection(collection_id) is None:
            msg = f"no collection {collection_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)
        found = require_owned(
            self.workspace,
            await store.collection_documents(collection_id, limit=limit, offset=offset),
        )
        return r.DocumentList(
            count=len(found),
            limit=limit,
            offset=offset,
            documents=tuple(_summary(document) for document in found),
        )

    # --- tags -----------------------------------------------------------------------------

    async def tag_create(self, name: str, *, color: str | None = None) -> r.TagSummary:
        """Create a tag, or return the existing one of that name. Idempotent by design.

        Raises:
            ValueError: The name is empty once normalised.
        """
        store = await self._backend.organisation()
        return _tag(await store.ensure_tag(name, color=color))

    async def tag_list(self) -> r.TagList:
        """Every tag in this workspace."""
        store = await self._backend.organisation()
        found = await store.list_tags()
        return r.TagList(count=len(found), tags=tuple(_tag(item) for item in found))

    async def tag_delete(self, tag_id: str) -> r.TagDeleted:
        """Delete a tag. Documents keep their other tags.

        Raises:
            UnknownEntityError: No such tag in this workspace.
        """
        store = await self._backend.organisation()
        await store.delete_tag(tag_id)
        return r.TagDeleted(tag_id=tag_id, deleted=True)

    async def document_tag(self, document_id: str, tag_ids: Sequence[str]) -> r.DocumentTags:
        """Apply tags to a document.

        Raises:
            UnknownEntityError: No such document or tag in this workspace.
        """
        await self._require_document(document_id)
        store = await self._backend.organisation()
        changed = await store.tag_document(document_id, list(tag_ids))
        return r.DocumentTags(
            document_id=document_id,
            changed=changed,
            tags=tuple(_tag(item) for item in await store.tags_for(document_id)),
        )

    async def document_untag(self, document_id: str, tag_ids: Sequence[str]) -> r.DocumentTags:
        """Remove tags from a document.

        Raises:
            UnknownEntityError: No such document in this workspace.
        """
        await self._require_document(document_id)
        store = await self._backend.organisation()
        changed = await store.untag_document(document_id, list(tag_ids))
        return r.DocumentTags(
            document_id=document_id,
            changed=changed,
            tags=tuple(_tag(item) for item in await store.tags_for(document_id)),
        )

    # --- the trash ------------------------------------------------------------------------

    async def document_trash(self, *, limit: int = 50, offset: int = 0) -> r.TrashList:
        """What is in the trash, longest-deleted first — the order the sweep will take them.

        Raises:
            CrossWorkspaceError: A document came back whose id was not minted here.
        """
        store = await self._backend.organisation()
        entries = await store.list_trash(
            grace_s=self.settings.ingest.soft_delete_grace_s, limit=limit, offset=offset
        )
        require_owned(self.workspace, [entry.document for entry in entries])
        return r.TrashList(
            count=len(entries),
            limit=limit,
            offset=offset,
            documents=tuple(
                r.TrashedDocument(
                    document=_summary(entry.document),
                    deleted_at=entry.deleted_at.isoformat(),
                    purged=entry.purged,
                    restorable_until=(
                        entry.restorable_until.isoformat() if entry.restorable_until else None
                    ),
                    free_restore=entry.free_restore,
                )
                for entry in entries
            ),
        )

    async def document_restore(self, document_id: str) -> r.DocumentRestored:
        """Take a document out of the trash, and say what that achieved.

        Raises:
            UnknownEntityError: Nothing was restored. The reason is in the message: no such
                document, or one that is not in the trash.
        """
        store = await self._backend.organisation()
        restoration = await store.restore_document(document_id)
        if not restoration.restored:
            raise UnknownEntityError(restoration.reason)
        return r.DocumentRestored(
            document_id=restoration.document_id,
            restored=True,
            needs_reparse=restoration.needs_reparse,
            reason=restoration.reason,
        )

    # --- the workbench --------------------------------------------------------------------

    async def workbench(self, document_id: str) -> r.Workbench:
        """One document as it was chunked, for seeing what retrieval actually indexes.

        Read-only and one document at a time. A chunking problem and a retrieval problem look
        identical from a ranked list, and this is the only view that tells them apart.

        Raises:
            UnknownEntityError: No live document with that id in this workspace.
        """
        detail = await self.document_get(document_id, chunks=True)
        return r.Workbench(
            document=detail.document,
            count=len(detail.chunks),
            tokens=sum(chunk.token_count for chunk in detail.chunks),
            blocks=tuple(
                r.WorkbenchBlock(
                    id=chunk.id,
                    position=chunk.position,
                    kind=chunk.kind,
                    heading_path=chunk.heading_path,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    anchor=chunk.anchor,
                )
                for chunk in detail.chunks
            ),
        )

    # --- administration -------------------------------------------------------------------

    async def query_logs(self, *, limit: int = 50, offset: int = 0) -> r.QueryLogPage:
        """A page of retrieval telemetry, newest first.

        Written by :meth:`search` and :meth:`ask_stream`, so the page describes every
        retrieval this installation ran rather than the ones one surface happened to make.
        """
        telemetry = await self._backend.telemetry()
        rows, total = await telemetry.query_logs(limit=limit, offset=offset)
        return r.QueryLogPage(
            total=total,
            count=len(rows),
            limit=limit,
            offset=offset,
            entries=tuple(
                r.QueryLogEntry(
                    id=_text(row.get("id")),
                    query=_text(row.get("query")),
                    profile=_text(row.get("profile")),
                    chunks=_int(row.get("chunks")),
                    confidence=_float_or_none(row.get("confidence")),
                    elapsed_ms=_int_or_none(row.get("elapsed_ms")),
                    created_at=_text(row.get("created_at")),
                )
                for row in rows
            ),
        )

    async def audit_log(
        self, *, limit: int = 50, offset: int = 0, event_type: str | None = None
    ) -> r.AuditPage:
        """A page of the audit trail, newest first.

        ``enabled`` is reported alongside the entries because an empty audit log means two
        different things — nothing happened, or nothing was recorded — and an operator reading
        one needs to know which.
        """
        telemetry = await self._backend.telemetry()
        rows, total = await telemetry.audit_logs(limit=limit, offset=offset, event_type=event_type)
        return r.AuditPage(
            enabled=self.settings.security.audit.enabled,
            total=total,
            count=len(rows),
            limit=limit,
            offset=offset,
            entries=tuple(
                r.AuditEntry(
                    id=_text(row.get("id")),
                    event_type=_text(row.get("event_type")),
                    actor=_text_or_none(row.get("actor")),
                    ip_address=_text_or_none(row.get("ip_address")),
                    details=_json_object(row.get("details")),
                    created_at=_text(row.get("created_at")),
                )
                for row in rows
            ),
        )

    async def search_quality(self) -> r.SearchQuality:
        """What the evaluation harness has recorded, rendered by the harness itself.

        This reports; it does not measure. :mod:`manicule.evaluation` is the only thing in
        this project that decides whether one retrieval configuration beats another, and a
        second scoring path reachable over HTTP would be a number nobody could reconcile with
        the one the harness produces.

        A report built from an **example** query set is returned with ``is_evidence`` false
        and the harness's own caveat attached. An example query set is an illustration of the
        instrument, and presenting one as a measurement is the failure the harness exists to
        prevent.
        """
        from manicule.evaluation.errors import EvaluationError  # noqa: PLC0415 - only here
        from manicule.evaluation.preference import PreferenceStore  # noqa: PLC0415
        from manicule.evaluation.report import ILLUSTRATIVE, build_report  # noqa: PLC0415

        path = self.settings.data_dir / "evaluation" / "preferences.jsonl"
        store = PreferenceStore(path)
        records = await asyncio.to_thread(lambda: list(store.records()))
        if not records:
            return r.SearchQuality(
                available=False,
                path=str(path),
                caveat=(
                    f"no preference judgements have been recorded at {path}. Retrieval quality "
                    f"is measured by running the pairwise harness against a query set; there "
                    f"is no number to report until somebody has judged some pairs."
                ),
            )
        try:
            report = await asyncio.to_thread(build_report, records)
        except EvaluationError as exc:
            return r.SearchQuality(
                available=True,
                path=str(path),
                records=len(records),
                caveat=(
                    f"{len(records)} judgement(s) were recorded and no report can be built "
                    f"from them: {exc}"
                ),
            )
        return r.SearchQuality(
            available=True,
            is_evidence=report.is_evidence,
            caveat="" if report.is_evidence else ILLUSTRATIVE,
            path=str(path),
            left_label=report.left_label,
            right_label=report.right_label,
            query_set=report.query_set,
            records=report.total,
            judged=report.judged,
            report=report.render(),
        )

    async def plugin_health(self) -> r.PluginHealthReport:
        """Installed plugins, with the health of whatever each one has constructed.

        A component that has not been built yet is ``unknown`` rather than ``ok``. The
        distinction is the whole value of the report: a plugin nothing has asked for has not
        been proved healthy, and saying it is would be a diagnostic that measured nothing.
        """
        discovery = self._backend.discovery
        checks = {check.name: check for check in await self._backend.component_checks()}
        listed = await self.plugin_list()
        health: list[r.PluginHealth] = []
        for plugin in listed.plugins:
            states = [
                checks[f"component:{component.kind}:{component.name}"]
                for component in plugin.components
                if f"component:{component.kind}:{component.name}" in checks
            ]
            worst: r.CheckState = "unknown" if not states else "ok"
            details: list[str] = []
            for check in states:
                if _severity(check.state) > _severity(worst):
                    worst = check.state
                if check.detail:
                    details.append(f"{check.name}: {check.detail}")
            health.append(
                r.PluginHealth(
                    name=plugin.name,
                    version=plugin.version,
                    enabled=plugin.enabled,
                    components=len(plugin.components),
                    state=worst,
                    detail=(
                        "; ".join(details)
                        or "nothing this plugin registered has been constructed yet"
                    ),
                )
            )
        return r.PluginHealthReport(
            count=len(health),
            plugins=tuple(health),
            disabled=listed.disabled if discovery is not None else (),
        )

    # --- identity -------------------------------------------------------------------------

    async def authenticate(self, secret: str) -> r.Identity:
        """Turn a presented API key into an identity, or say plainly that it is not one.

        The whole decision lives here rather than in the surface that received the header, so
        the CLI, the MCP server and the HTTP API cannot disagree about what a valid key is.
        A key is only ever usable in the workspace it was minted for: the store is scoped, and
        a key from another tenant simply does not resolve.

        When ``security.auth.mode`` is ``none`` this reports an unauthenticated identity
        rather than inventing one — and it is the bind policy, not this method, that stops an
        unauthenticated installation being reachable from anywhere but loopback.
        """
        mode = self.settings.security.auth.mode
        if mode is AuthMode.NONE:
            return r.Identity(
                authenticated=False, mode=mode.value, role="", workspace=self.workspace
            )
        if not secret:
            return r.Identity(
                authenticated=False, mode=mode.value, role="", workspace=self.workspace
            )
        keys = await self._backend.keys()
        summary = await keys.verify(secret)
        if summary is None:
            return r.Identity(
                authenticated=False, mode=mode.value, role="", workspace=self.workspace
            )
        return r.Identity(
            authenticated=True,
            mode=mode.value,
            role=summary.role,
            key_id=summary.id,
            key_name=summary.name,
            workspace=summary.workspace,
        )

    async def auth_providers(self) -> r.AuthProviders:
        """Which identity providers are configured, by name and type only.

        Never a client secret and never a redirect that was not configured. When the mode is
        not ``oauth`` the list is empty and ``detail`` says why, rather than advertising a
        login route that would refuse every attempt.
        """
        auth = self.settings.security.auth
        if auth.mode is not AuthMode.OAUTH:
            return r.AuthProviders(
                mode=auth.mode.value,
                count=0,
                detail=(
                    f"security.auth.mode is {auth.mode.value!r}, so no interactive login is "
                    f"offered. Authenticate with an API key, or configure OAuth."
                ),
            )
        return r.AuthProviders(
            mode=auth.mode.value,
            count=len(auth.providers),
            providers=tuple(provider.type for provider in auth.providers),
        )

    # --- helpers --------------------------------------------------------------------------

    async def _require_document(self, document_id: str) -> None:
        """Prove a document is live and this tenant's before anything is attached to it."""
        store = await self._backend.documents()
        document = await store.get_document(document_id)
        if document is None:
            msg = f"no live document {document_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)
        require_owns(self.workspace, document)

    async def _audit(self, event_type: str, *, details: Mapping[str, object]) -> None:
        """Record one security-relevant event, when auditing is switched on.

        Gated here rather than in the writer, so that "auditing is off" is a decision made
        once against configuration instead of a condition every call site repeats — and the
        admin surface reports the same switch alongside the entries, so an empty trail is
        never mistaken for a quiet one.
        """
        audit = self.settings.security.audit
        if not audit.enabled:
            return
        if audit.events and event_type not in audit.events:
            return
        telemetry = await self._backend.telemetry()
        # Deliberately **not** wrapped the way `_record_query` is. An audit entry that cannot
        # be written must fail the operation it was auditing: a trail with holes in it is worse
        # than none, because the holes are invisible and the operation reported success.
        await telemetry.record_audit(event_type, details=details)

    async def _record_query(
        self, query: Query, retrieved: RetrievalResult, *, started: float
    ) -> None:
        """Write one retrieval into ``query_logs``.

        In the service rather than in a surface: this is the row the admin surface pages
        through, and telemetry that only records the traffic of whichever surface remembered to
        write it describes nothing.

        **A failure here does not fail the query.** Retrieval is a read; recording it is a
        write, and on SQLite that write can lose to a lock or a busy timeout. Letting it
        propagate would mean a search that worked yesterday returns 500 today because an
        *observability* insert could not get the writer — a read made conditional on a write.
        It is logged at warning rather than swallowed, so a telemetry backend that has stopped
        working is visible instead of merely quiet.
        """
        try:
            telemetry = await self._backend.telemetry()
            confidence = retrieved.confidence
            await telemetry.record_query(
                query.text,
                profile=query.profile.value,
                chunk_ids=[candidate.chunk.id for candidate in retrieved.context.passages],
                confidence=confidence.score if confidence else None,
                elapsed_ms=_millis(started),
            )
        except (ManiculeError, ValueError, OSError) as exc:
            # The query text is deliberately not in the message. It is user content, and a
            # log line is exactly where `security.storage.redact_logs_content` says it must
            # not appear.
            _log.warning("could not record query telemetry: %s: %s", type(exc).__name__, exc)

    def _query(
        self,
        text: str,
        *,
        limit: int,
        profile: str | None,
        sources: Sequence[str] = (),
        media_types: Sequence[str] = (),
    ) -> Query:
        """Build a query already carrying this workspace.

        The filter has no default workspace and cannot be built without one, so there is no
        path from here to an unscoped search.
        """
        stripped = text.strip()
        if not stripped:
            msg = "a query cannot be empty"
            raise ConfigError(msg)
        chosen = self.settings.rag.profile
        if profile is not None:
            try:
                chosen = RetrievalProfile(profile)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in RetrievalProfile)
                msg = f"no such profile {profile!r}. Available: {allowed}"
                raise ConfigError(msg) from exc
        return Query(
            text=stripped,
            limit=limit,
            profile=chosen,
            filter=Filter(
                workspace_ids=frozenset({self.workspace}),
                sources=frozenset(sources),
                media_types=frozenset(media_types),
            ),
        )

    async def _require_scoped_context(self, retrieved: RetrievalResult) -> None:
        """Prove every retrieved passage belongs to this workspace, before anything leaves."""
        await self._require_scoped_chunks(
            candidate.chunk for candidate in retrieved.context.passages
        )

    async def _require_scoped_chunks(self, chunks: Iterable[Chunk]) -> dict[str, Document]:
        """Resolve each chunk's document through the scoped store and check its identity.

        Two independent facts have to hold, and neither implies the other: the store must be
        able to see the document at all — the lookup is workspace-scoped and skips the trash —
        and the document's id must be a digest of *this* workspace. A store that ignored its
        scope passes the first and fails the second.

        Raises:
            CrossWorkspaceError: A chunk points at a document this workspace does not own.
        """
        wanted = list(dict.fromkeys(chunk.document_id for chunk in chunks))
        if not wanted:
            return {}
        store = await self._backend.documents()
        # One query rather than one per document. `document_ids` is a field the store honours,
        # so this is the same scoped, trash-excluding lookup — and a ranked page routinely
        # spans several documents, which made the per-document form N round trips on the hot
        # path of every search and every answer.
        asked = frozenset(wanted)
        page = await store.list_documents(
            Filter(workspace_ids=frozenset({self.workspace}), document_ids=asked),
            limit=len(wanted),
        )
        # Restricted to what was asked for. A store that returns more than the filter allowed
        # is a defect its own conformance suite owns; here the question is only whether every
        # requested document came back, and whether each one belongs to this tenant.
        found: dict[str, Document] = {
            document.id: document for document in page if document.id in asked
        }
        missing = [document_id_ for document_id_ in wanted if document_id_ not in found]
        if missing:
            msg = (
                f"retrieval returned {len(missing)} chunk(s) whose document workspace "
                f"{self.workspace!r} cannot see. Nothing was returned: a search that quietly "
                f"drops the rows it should not have had is a search that leaked the ranking."
            )
            raise CrossWorkspaceError(msg)
        require_owned(self.workspace, found.values())
        return found

    def _answer_payload(
        self,
        question: str,
        envelope: AnswerEnvelope,
        aside: AskAside,
        started: float,
        conversation_id: str | None,
    ) -> r.AnswerResultPayload:
        return r.AnswerResultPayload(
            question=question,
            text=envelope.text,
            citations=tuple(
                r.AnswerCitation(
                    slot=citation.slot,
                    document_id=citation.document_id,
                    chunk_id=citation.chunk_id,
                    uri=citation.uri,
                    title=citation.title,
                    heading_path=citation.heading_path,
                    kind=citation.kind.value,
                    anchor=_json_object(citation.anchor.model_dump(mode="json")),
                    quote=citation.quote,
                    verification=citation.verification.value,
                )
                for citation in envelope.citations
            ),
            dropped=len(envelope.dropped),
            confidence=envelope.confidence,
            confidence_band=aside.confidence_band,
            corpus_consulted=envelope.corpus_consulted,
            ungrounded=envelope.ungrounded,
            context_truncated=envelope.context_truncated,
            redacted=envelope.redacted,
            finish_reason=envelope.finish_reason.value if envelope.finish_reason else None,
            error=envelope.error,
            conversation_id=conversation_id,
            message_id=aside.message_id,
            model=self.settings.llm.model,
            elapsed_ms=_millis(started),
        )


# --- module helpers --------------------------------------------------------------------------


def hardware() -> dict[str, JsonValue]:
    """What this machine is, as far as it can be probed without a dependency.

    Used by ``init`` to pick the embedding backend. Apple Silicon runs MLX in-process; every
    other machine runs onnxruntime. Both produce the same vectors within the parity tolerance
    the embedding suite asserts — the platform changes throughput, never what lands in the
    index.
    """
    machine = platform.machine()
    system = platform.system()
    apple_silicon = system == "Darwin" and machine == "arm64"
    probe: dict[str, JsonValue] = {
        "system": system,
        "machine": machine,
        "python": platform.python_version(),
        "apple_silicon": apple_silicon,
        "cpus": _cpu_count(),
    }
    memory = _total_memory_bytes()
    if memory is not None:
        probe["memory_bytes"] = memory
        # Unified memory on Apple Silicon is shared with the GPU, so the figure that matters
        # for a model is the whole of it rather than a separate video allocation.
        probe["unified_memory"] = apple_silicon
    return probe


def _cpu_count() -> int:
    import os  # noqa: PLC0415 - only the probe needs it

    return os.cpu_count() or 1


def _total_memory_bytes() -> int | None:
    import os  # noqa: PLC0415 - only the probe needs it

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform dependent
        return None
    if pages < 0 or size < 0:  # pragma: no cover - platform dependent
        return None
    return pages * size


async def _fetch_registry(
    url: str, installed: set[str]
) -> tuple[tuple[r.AvailablePlugin, ...], str]:
    """Read the community plugin list, and say plainly when it could not be read.

    The HTTP client is imported here rather than at module scope, so a machine that never
    browses the registry never loads one — the same rule every other optional dependency in
    this project follows.
    """
    try:
        import httpx  # noqa: PLC0415 - optional, and only this function needs it
    except ImportError:
        return (), (
            "no HTTP client is installed, so the registry cannot be read. Install "
            "manicule[connectors] to add one."
        )
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.text
    except Exception as exc:  # noqa: BLE001 - a registry that cannot be read is not a crash
        return (), f"{type(exc).__name__}: {exc}"

    try:
        listing = _RegistryListing.model_validate_json(body)
    except ValidationError as exc:
        return (), f"the registry at {url} is not a plugin listing this build can read: {exc}"

    offered: list[r.AvailablePlugin] = []
    for entry in listing.plugins:
        supported, reason = _supports_this_core(entry.core_version)
        offered.append(
            r.AvailablePlugin(
                name=entry.name,
                version=entry.version,
                core_version=entry.core_version,
                summary=entry.summary,
                package=entry.package,
                url=entry.url,
                installed=entry.name in installed,
                compatible=supported,
                incompatible_reason=reason,
            )
        )
    return tuple(sorted(offered, key=lambda item: item.name)), ""


class _RegistryPlugin(BaseModel):
    """One entry in the community listing.

    ``extra="ignore"``, unlike everything else in this project, and the asymmetry is the
    point: this document is written by other people and served over the network, so a field
    added upstream must not make the whole listing unreadable here. What manicule *emits* is
    closed; what it *reads* from a third party is not.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str = Field(min_length=1)
    version: str = ""
    core_version: str = ""
    summary: str = ""
    package: str = ""
    url: str = ""


class _RegistryListing(BaseModel):
    """The community plugin listing."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    plugins: tuple[_RegistryPlugin, ...] = ()


def _supports_this_core(specifier: str) -> tuple[bool, str]:
    """Whether a registry entry declares support for the core that is running.

    Checked **before** anything is offered as installable, rather than at load time. A plugin
    installed and then refused at the next start is a broken installation discovered by
    restarting, which is the worst moment to find out.
    """
    if not specifier:
        return False, "declares no core_version, so nothing says which manicule it supports"
    from packaging.specifiers import InvalidSpecifier, SpecifierSet  # noqa: PLC0415
    from packaging.version import Version  # noqa: PLC0415

    try:
        allowed = SpecifierSet(specifier)
    except InvalidSpecifier:
        return False, f"core_version {specifier!r} is not a valid PEP 440 specifier"
    if allowed.contains(Version(CORE_VERSION), prereleases=True):
        return True, ""
    return False, f"requires manicule {specifier}, but {CORE_VERSION} is running"


def _installed_plugins(discovery: Discovery | None) -> set[str]:
    """Every plugin the environment provides, enabled or not.

    Disabled ones are included deliberately: ``plugin add`` re-enables one that is installed
    and switched off, and reporting it as "not installed" would send an operator to install a
    distribution they already have.
    """
    if discovery is None:
        return set()
    return {*discovery.names, *discovery.disabled}


def _connector_kind() -> Any:  # noqa: ANN401 - imported lazily to keep the module import light
    from manicule.plugins.manifest import ComponentKind  # noqa: PLC0415

    return ComponentKind.CONNECTOR


def _no_conversation(conversation_id: str, workspace: str) -> UnknownEntityError:
    """The one refusal every conversation write shares.

    One message rather than five, because they are all the same fact: an update scoped to this
    workspace matched no row. It never says whether the id exists in another tenant, because
    that answer is itself a cross-workspace disclosure.
    """
    return UnknownEntityError(
        f"no live conversation {conversation_id!r} in workspace {workspace!r}"
    )


def _conversation(record: ConversationRecord) -> r.ConversationSummary:
    return r.ConversationSummary(
        id=record.id,
        title=record.title,
        shared=record.shared,
        shared_at=record.shared_at.isoformat() if record.shared_at else None,
        share_expires_at=(record.share_expires_at.isoformat() if record.share_expires_at else None),
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        messages=record.messages,
    )


def _citation(citation: Citation) -> r.AnswerCitation:
    """One stored citation, in the shape every surface reports citations in."""
    return r.AnswerCitation(
        slot=citation.slot,
        document_id=citation.document_id,
        chunk_id=citation.chunk_id,
        uri=citation.uri,
        title=citation.title,
        heading_path=citation.heading_path,
        kind=citation.kind.value,
        anchor=_json_object(citation.anchor.model_dump(mode="json")),
        quote=citation.quote,
        verification=citation.verification.value,
    )


def _collection(collection: Collection) -> r.CollectionSummary:
    return r.CollectionSummary(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        rule=(
            _json_object(collection.rule.model_dump(mode="json"))
            if collection.rule is not None
            else None
        ),
        created_at=collection.created_at.isoformat(),
    )


def _tag(tag: Tag) -> r.TagSummary:
    return r.TagSummary(id=tag.id, name=tag.name, color=tag.color)


def _summary(document: Document, *, chunk_count: int | None = None) -> r.DocumentSummary:
    return r.DocumentSummary(
        id=document.id,
        source=document.source,
        source_id=document.source_id,
        uri=document.uri,
        title=document.title,
        media_type=document.media_type,
        status=document.status.value,
        status_detail=document.status_detail,
        failed_stage=document.failed_stage.value if document.failed_stage else None,
        content_hash=document.content_hash,
        chunk_count=chunk_count,
    )


def _ingest_payload(report: RunReport, started: float) -> r.IngestReport:
    return r.IngestReport(
        connector=report.connector,
        discovered=report.discovered,
        ingested=report.indexed,
        skipped=report.skipped_version + report.skipped_hash,
        failed=report.by_status.get(DocumentStatus.FAILED.value, 0),
        expanded=report.expanded,
        by_status=dict(report.by_status),
        error=report.error,
        elapsed_ms=_millis(started),
    )


def _millis(started: float) -> int:
    return max(int((time.monotonic() - started) * 1000), 0)


def _severity(state: r.CheckState) -> int:
    return {"ok": 0, "degraded": 1, "failing": 2, "unknown": 1}.get(state, 1)


def _parse_value(value: str) -> JsonValue:
    """JSON when it parses, the string when it does not.

    ``true``, ``12`` and ``["a"]`` mean what they look like; ``qwen2.5:14b`` is a string
    rather than a syntax error, which is the far commoner case at a terminal.
    """
    try:
        parsed: JsonValue = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed


def _local(path: Path | str) -> Path:
    """A user-supplied path, with ``~`` expanded.

    A module function rather than an inline call so that expansion — which reads the
    environment and the password database, never the filesystem — is not mistaken for the
    blocking I/O that has to leave the event loop.
    """
    return Path(path).expanduser()


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _update_config(path: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Read the config file, change it, validate the **whole** tree, then write it.

    Blocking, and called through :func:`asyncio.to_thread` by everything that edits
    configuration. Validation is not a formality here: it happens before the write, with the
    edited document passed as overrides, so a setting that is valid alone and contradicts
    another one never reaches the file. A config file that fails at the next start is a
    manicule that does not start.

    The file is rewritten rather than patched, so hand-written comments do not survive — the
    same trade :func:`manicule.config.loader.save_settings` makes, for the same reason.
    """
    document = _read_toml(path)
    mutate(document)
    load_settings(**document).require_valid()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    # The file can hold nothing secret by construction — `config set` refuses a credential —
    # but it names data directories and endpoints, and 0600 is what the rest of the data
    # directory uses.
    path.chmod(0o600)
    return document


def _plugin_mutation(name: str, *, enabled: bool) -> Callable[[dict[str, Any]], None]:
    """The edit that enables or disables one plugin.

    Disabling writes the name into ``plugins.disabled`` and never uninstalls anything.
    Enabling removes it from that list **and** adds it to ``plugins.enabled`` when an
    allowlist is in force — dropping it from the deny list alone would leave a plugin that
    reads as enabled and is still filtered out by the allowlist.
    """

    def mutate(document: dict[str, Any]) -> None:
        plugins: object = document.setdefault("plugins", {})
        if not isinstance(plugins, dict):  # pragma: no cover - a hand-edited file
            msg = "the config file's [plugins] section is not a table"
            raise ConfigError(msg)
        section = cast("dict[str, Any]", plugins)
        denied = [str(item) for item in _as_list(section.get("disabled"))]
        if enabled:
            section["disabled"] = [item for item in denied if item != name]
        elif name not in denied:
            section["disabled"] = [*denied, name]
        allowed = section.get("enabled")
        if isinstance(allowed, list) and enabled:
            listed = [str(item) for item in allowed]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            if name not in listed:
                section["enabled"] = [*listed, name]

    return mutate


def _initial_configuration(settings: Settings, provider: str) -> Callable[[dict[str, Any]], None]:
    """The edit ``init`` makes: a backend this machine can run, and a bind that is loopback."""

    def mutate(document: dict[str, Any]) -> None:
        embedding = cast("dict[str, Any]", document.setdefault("embedding", {}))
        embedding["provider"] = provider
        embedding.setdefault("model", settings.embedding.model)
        document.setdefault("workspace", settings.workspace)
        document.setdefault("data_dir", str(settings.data_dir))
        security = cast("dict[str, Any]", document.setdefault("security", {}))
        transport = cast("dict[str, Any]", security.setdefault("transport", {}))
        # Written out rather than left to the default. An operator reading their own config
        # file should be able to see that the bind is loopback without having to know what
        # manicule would have chosen for them.
        transport.setdefault("bind_host", "127.0.0.1")

    return mutate


def _assign(document: dict[str, Any], parts: Sequence[str], value: JsonValue) -> None:
    """Set one dotted key, creating the tables on the way to it."""
    cursor: dict[str, Any] = document
    for part in parts[:-1]:
        existing: object = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = cast("dict[str, Any]", existing)
    cursor[parts[-1]] = value


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []  # pyright: ignore[reportUnknownArgumentType]


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _json_object(value: object) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]


__all__ = ["DEFAULT_SOURCE", "ApplicationService", "AskAside", "hardware"]
