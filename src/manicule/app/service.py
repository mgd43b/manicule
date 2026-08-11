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
import os
import platform
import time
import tomllib
from dataclasses import dataclass
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

    from manicule.app.ports import Backend
    from manicule.core.content import Chunk, Document
    from manicule.generation.answers import AnswerEnvelope, AnswerEvent
    from manicule.ingest.pipeline import RunReport
    from manicule.plugins.registry import Discovery
    from manicule.retrieval.retriever import RetrievalResult

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
        envelope: AnswerEnvelope | None = None
        aside = AskAside()
        started = time.monotonic()
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
            if event.envelope is not None:
                envelope = event.envelope
        if envelope is None:  # pragma: no cover - the answer path always ends with `final`
            msg = "the answer stream ended without a final event"
            raise ManiculeError(msg)
        return self._answer_payload(question, envelope, aside, started, conversation_id)

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

        query = self._query(question, limit=limit or 8, profile=profile, sources=sources)
        retriever = await self._backend.retriever()
        retrieved = await retriever.retrieve(query)
        await self._require_scoped_context(retrieved)

        answerer = await self._backend.answerer()
        confidence = retrieved.confidence
        if aside is not None and confidence is not None:
            aside.confidence_band = confidence.band.value
        request = AnswerRequest(
            query=query,
            context=retrieved.context,
            conversation_id=conversation_id,
            confidence=confidence.score if confidence else None,
            corpus_consulted=retrieved.cites_the_corpus,
        )
        result = AnswerResult()
        try:
            async with answering(answerer, request, result) as events:
                async for event in events:
                    yield event
        finally:
            if aside is not None:
                aside.message_id = result.message_id

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

    async def plugin_list(self, *, registry: bool = False) -> r.PluginList:
        """What is installed, and — when configuration allows it — what is offered."""
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
        return r.ApiKeyRevoked(id=summary.id, name=summary.name, revoked=True)

    # --- operations the command line owns -------------------------------------------------

    async def backup(self, target: Path | str) -> r.BackupReport:
        """Take a consistent copy of the data directory."""
        maintenance = await self._backend.maintenance()
        manifest = await maintenance.backup(_local(target))
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

    # --- helpers --------------------------------------------------------------------------

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


def _json_object(value: object) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]


__all__ = ["DEFAULT_SOURCE", "ApplicationService", "AskAside", "hardware"]
