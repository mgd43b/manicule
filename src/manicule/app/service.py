"""The application service: every operation manicule offers, once.

Both surfaces are adapters over this class and neither contains any behavior of its own.
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
import contextlib
import hashlib
import json
import logging
import os
import platform
import secrets
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from manicule.app import results as r
from manicule.app.tenancy import CrossWorkspaceError, require_owned, require_owns
from manicule.config.loader import load_settings
from manicule.config.settings import (
    AuthMode,
    BrowserProvider,
    ConnectorSettings,
    Role,
    Settings,
    config_file,
    looks_secret,
)
from manicule.connectors.enriched import ENRICHED_KEY, AdapterOutcome
from manicule.container import keys
from manicule.core.content import PREVIOUS_IDENTITY, DocumentStatus
from manicule.core.errors import (
    ConfigError,
    ManiculeError,
    PolicyError,
    UnknownComponentError,
    UnknownEntityError,
)
from manicule.core.glossary import GlossaryEntry, QueryExpansion
from manicule.core.ids import document_id
from manicule.core.rebuild import (
    RebuildLeaseConflictError,
    RebuildLeaseError,
    RebuildOperationError,
    RebuildPublicationValidationError,
    RebuildStorageError,
    RebuildTerminalError,
    RebuildTerminalGenerationError,
    RebuildValidationError,
)
from manicule.core.retrieval import Filter, Query, RetrievalProfile
from manicule.core.source_lifecycle import LifecycleOutcome, LifecyclePlan
from manicule.core.version import CORE_VERSION
from manicule.ingest.reindex import DEFAULT_SWEEP_BATCH

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from pydantic import SecretStr

    from manicule.app.ports import Backend, Conversing
    from manicule.connectors.browser import BrowserSessionProvider
    from manicule.connectors.config import ConfluenceConfig
    from manicule.connectors.enriched import EnrichedProfile
    from manicule.connectors.filesystem import FilesystemConnector
    from manicule.connectors.sessions import SessionStore
    from manicule.core.acquisition import AcquisitionRun
    from manicule.core.content import Chunk, Document
    from manicule.core.organization import Collection, CollectionRule, Tag
    from manicule.core.rebuild import RebuildCheckpoint, RebuildEstimate
    from manicule.core.retrieval import Confidence
    from manicule.embedding.artifacts import WeightsPlan
    from manicule.generation.answers import AnswerEnvelope, AnswerEvent, Citation
    from manicule.generation.ports import ConversationRecord
    from manicule.ingest.pipeline import RunReport, Watching
    from manicule.ingest.reembed import ReembedRecovery, ReembedRun
    from manicule.parsers.config import SourceCodeConfig
    from manicule.plugins.registry import Discovery
    from manicule.retrieval.retriever import RetrievalResult

_log = logging.getLogger("manicule.app")
"""For the one thing here that is reported rather than raised: a telemetry write that failed.

A module logger rather than a print or a swallowed exception, so an operator who has configured
logging sees it and one who has not is not spammed by a library.
"""

_LANGUAGES_NAMED = 6
"""How many missing grammars ``doctor`` names before it counts the rest.

Named at all because "seed the grammars" without saying which is a fix nobody can check;
bounded because all twenty-four of them is a paragraph, and the sentence that says what to do
was being lost inside it.
"""

_IMMUTABLE_COMMIT_LENGTH = 40

_COUNT_PAGE = 500
"""How many documents a counting or sweeping walk reads per round trip.

A page size, deliberately not a limit: both callers loop until the corpus ends, so this
changes how many statements a count costs and never what it reports. A cap here would make
``collection_counts`` report a floor and ``collection_orphans`` claim a cleanup it had only
partly done.
"""

DEFAULT_SOURCE = "local"
"""The source name ``index_path`` uses when none is given.

A constant rather than something derived from the directory, because a document's identity is
``(workspace, source, source_id)``: a source name that changed with the working directory
would re-index the same file as a new document every time it was indexed from somewhere else.
"""

_CONFLICTING = 2
"""How many readable definitions make a disagreement.

Named rather than written as a literal because the number *is* the definition of the word: one
definition is a definition, and a "conflict" reported with a single candidate is a warning a
reader cannot act on.
"""


def _reembed_run_report(run: ReembedRun) -> r.ReembedRunReport:
    """Public aggregate projection of a private durable re-embedding commitment."""
    from manicule.ingest.reembed import ReembedState  # noqa: PLC0415

    plan = run.commitment.plan
    remaining_chunks = max(0, plan.chunks - run.workspace_chunks_completed)
    seconds_per_chunk = plan.estimated_seconds / plan.chunks if plan.chunks else 0.0
    terminal = run.state in {
        ReembedState.PUBLISHED,
        ReembedState.SUPERSEDED,
        ReembedState.FAILED,
    }
    target_identity = hashlib.sha256(run.commitment.target_fingerprint.encode("utf-8")).hexdigest()
    failed = run.state in {ReembedState.SUPERSEDED, ReembedState.FAILED}
    lifecycle_outcome: r.LifecycleOutcome = (
        "complete" if run.state is ReembedState.PUBLISHED else "failed" if failed else "running"
    )
    lifecycle_phase: r.LifecyclePhase = (
        "complete" if run.state is ReembedState.PUBLISHED else "failed" if failed else "reembedding"
    )
    return r.ReembedRunReport(
        run_id=run.id,
        state=run.state.value,
        documents=plan.documents,
        chunks=plan.chunks,
        documents_completed=run.workspace_documents_completed,
        chunks_completed=run.workspace_chunks_completed,
        estimated_remaining_seconds=remaining_chunks * seconds_per_chunk,
        retry_required=not terminal,
        terminal=terminal,
        published=run.state is ReembedState.PUBLISHED,
        target_identity=target_identity,
        target_dimension=run.commitment.target_dimension,
        lifecycle=r.LifecycleProgress(
            phase=lifecycle_phase,
            outcome=lifecycle_outcome,
            enumerated_items=plan.chunks,
            acquired_items=run.workspace_chunks_completed,
            pending_items=0 if terminal else remaining_chunks,
            source_generation_identity=hashlib.sha256(
                run.commitment.snapshot.live.generation_id.encode("utf-8")
            ).hexdigest(),
            derived_generation_identity=target_identity,
            can_continue_offline=True,
            estimated_remaining_items=0 if terminal else remaining_chunks,
            estimated_remaining_seconds=(0 if terminal else remaining_chunks * seconds_per_chunk),
        ),
    )


def _snapshot_status_report(
    run: AcquisitionRun, verified: bool, *, verification_performed: bool
) -> r.SnapshotStatusReport:
    """Project a private manifest onto the aggregate surface contract."""
    from manicule.core.acquisition import AcquisitionRunState  # noqa: PLC0415

    derivation_pending = max(0, run.acquired_count - run.indexed_count)
    omission_pending = (
        run.omission_count
        if run.completeness is not None and run.completeness.value == "partial"
        else 0
    )
    pending = derivation_pending + omission_pending
    corrupt = verification_performed and run.acquisition_completed_at is not None and not verified
    terminal = run.state is AcquisitionRunState.SETTLED and omission_pending == 0
    phase: r.LifecyclePhase = (
        "failed"
        if corrupt
        else "complete"
        if terminal
        else "acquiring"
        if omission_pending
        else "rebuilding"
        if run.state is AcquisitionRunState.INDEXING
        else "acquiring"
    )
    now = datetime.now(UTC)
    # "Running" is a claim about a live worker, so it is read off the lease rather than off
    # the absence of bad news.
    #
    # A worker that lost its run does not get to update the row on its way out — that is what
    # losing it means — so an unfinished run with no diagnostic used to project ``running``
    # whether a process was working on it or nothing had touched it for an hour. The two look
    # identical from here and are opposite operationally: one is progress, the other is
    # resumable work that nobody has resumed. An operator who cannot tell them apart waits.
    #
    # The lease answers it directly. It is renewed at a third of its lifetime for exactly as
    # long as a worker is alive, so an expired or unheld one is a run with no owner — which is
    # ``incomplete``: unfinished, inactive, and resumable from what is already committed. That
    # is also true of a cleanly released lease after a canceled run, which is the same
    # situation arrived at politely.
    held = (
        run.lease_owner is not None
        and run.lease_expires_at is not None
        and run.lease_expires_at > now
    )
    outcome: r.LifecycleOutcome = (
        "failed"
        if corrupt
        else "complete"
        if terminal
        else "incomplete"
        if run.diagnostic is not None or omission_pending or not held
        else "running"
    )
    age = (
        float(max(0, int((now - run.updated_at).total_seconds())))
        if pending and not terminal and not corrupt
        else None
    )
    return r.SnapshotStatusReport(
        connector=run.connector,
        snapshot_id=run.id,
        state=run.state.value,
        verified=verified,
        verification_performed=verification_performed,
        full_inventory_authority=_public_full_inventory_authority(run.full_inventory_authority),
        lifecycle=r.LifecycleProgress(
            phase=phase,
            outcome=outcome,
            enumerated_items=run.discovered_count,
            acquired_items=run.acquired_count,
            reused_items=run.reused_count,
            omitted_items=run.omission_count,
            failed_items=run.retry_count,
            pending_items=0 if terminal or corrupt else pending,
            snapshot_completeness=run.completeness.value if run.completeness else "",
            reproducibility_policy=run.promotion_policy.value,
            snapshot_identity=run.id,
            snapshot_promoted=run.promoted_at is not None,
            source_generation_identity=run.membership_hash or "",
            candidate_watermark_present=run.candidate_watermark is not None,
            committed_watermark_present=run.watermark_committed_at is not None,
            backlog_items=0 if terminal or corrupt else pending,
            backlog_bytes=0 if terminal or corrupt else run.acquired_blob_bytes,
            oldest_backlog_age_seconds=age,
            can_continue_offline=(
                run.promoted_at is not None
                and bool(run.membership_hash)
                and (verified or not verification_performed)
                and omission_pending == 0
            ),
            estimated_remaining_items=0 if terminal or corrupt else pending,
            inventory_recovery=(
                "" if run.inventory_state.value == "current" else run.inventory_state.value
            ),
            reconciled_deleted_items=run.reconciled_deleted_count,
            full_inventory_authority=_public_full_inventory_authority(run.full_inventory_authority),
        ),
    )


def _lifecycle_plan_report(plan: LifecyclePlan) -> r.LifecycleReport:
    phase: r.LifecyclePhase = (
        "resetting"
        if plan.operation.value == "reset_derived"
        else "deleting"
        if plan.operation.value in {"release_source_history", "delete_snapshot"}
        else "rebuilding"
    )
    return r.LifecycleReport(
        operation=plan.operation.value,
        dry_run=True,
        eligible_items=plan.eligible_items,
        eligible_bytes=plan.eligible_bytes,
        protected_items=plan.protected_items,
        protected_bytes=plan.protected_bytes,
        snapshot_items=plan.snapshot_items,
        unrecoverable_items=plan.unrecoverable_items,
        unrecoverable_bytes=plan.unrecoverable_bytes,
        confirmation=plan.confirmation,
        lifecycle=r.LifecycleProgress(
            phase=phase,
            outcome="deferred",
            dry_run=True,
            enumerated_items=plan.eligible_items + plan.protected_items,
            pending_items=plan.eligible_items,
            snapshot_completeness="complete" if plan.snapshot_items else "",
            snapshot_promoted=True if plan.snapshot_items else None,
            backlog_items=plan.eligible_items,
            backlog_bytes=plan.eligible_bytes,
            can_continue_offline=plan.snapshot_items > 0,
            estimated_remaining_items=plan.eligible_items,
        ),
    )


def _lifecycle_outcome_report(outcome: LifecycleOutcome) -> r.LifecycleReport:
    phase: r.LifecyclePhase = (
        "resetting"
        if outcome.operation.value == "reset_derived"
        else "deleting"
        if outcome.operation.value in {"release_source_history", "delete_snapshot"}
        else "rebuilding"
    )
    return r.LifecycleReport(
        operation=outcome.operation.value,
        dry_run=False,
        removed_items=outcome.removed_items,
        released_bytes=outcome.released_bytes,
        snapshot_items=outcome.snapshot_items,
        source_contacted=outcome.source_contacted,
        documents_retired=outcome.documents_retired,
        chunks_removed=outcome.chunks_removed,
        memberships_removed=outcome.memberships_removed,
        vector_rows_removed=outcome.vector_rows_removed,
        publications_removed=outcome.publications_removed,
        generations_terminalized=outcome.generations_terminalized,
        vector_store_removed=outcome.vector_store_removed,
        fingerprints_cleared=outcome.fingerprints_cleared,
        runtime_cache_invalidated=outcome.runtime_cache_invalidated,
        lifecycle=r.LifecycleProgress(
            phase=phase,
            outcome="complete",
            acquired_items=outcome.removed_items,
            snapshot_completeness="complete" if outcome.snapshot_items else "",
            snapshot_promoted=True if outcome.snapshot_items else None,
            can_continue_offline=outcome.snapshot_items > 0,
        ),
    )


def _rebuild_plan_report(estimate: RebuildEstimate) -> r.RebuildPlanReport:
    refusal = (
        r.LifecycleRefusal(code=estimate.refusal.value, count=estimate.missing_count)
        if estimate.refusal is not None
        else None
    )
    return r.RebuildPlanReport(
        generation_id=estimate.generation_id,
        snapshot_id=estimate.snapshot_run_id,
        documents=estimate.documents,
        known_source_bytes=estimate.known_source_bytes,
        estimated_chunks=estimate.estimated_chunks,
        estimated_seconds=estimate.estimated_seconds,
        estimated_peak_memory_bytes=estimate.estimated_peak_memory_bytes,
        estimated_temporary_bytes=estimate.estimated_temporary_bytes,
        missing_count=estimate.missing_count,
        refusal_code=estimate.refusal.value if estimate.refusal else None,
        runnable=estimate.runnable,
        current_chunk_fingerprint=estimate.current_chunk_fingerprint,
        target_chunk_fingerprint=estimate.target_chunk_fingerprint,
        over_budget_chunks=estimate.over_budget_chunks,
        max_stored_chunk_tokens=estimate.max_stored_chunk_tokens,
        estimated_embedding_chunks=estimate.estimated_embedding_chunks,
        network_required=estimate.network_required,
        lifecycle=r.LifecycleProgress(
            phase="rebuilding",
            outcome="deferred" if estimate.runnable else "refused",
            dry_run=True,
            enumerated_items=estimate.documents,
            failed_items=estimate.missing_count,
            pending_items=estimate.documents,
            snapshot_completeness="complete" if estimate.runnable else "",
            snapshot_identity=estimate.snapshot_run_id,
            snapshot_promoted=estimate.runnable,
            source_generation_identity=estimate.snapshot_run_id,
            derived_generation_identity=estimate.generation_id,
            backlog_items=estimate.documents,
            backlog_bytes=estimate.known_source_bytes,
            can_continue_offline=estimate.runnable,
            estimated_remaining_items=estimate.documents,
            estimated_remaining_seconds=estimate.estimated_seconds,
            refusal=refusal,
        ),
    )


def _rebuild_run_report(checkpoint: RebuildCheckpoint) -> r.RebuildRunReport:
    terminal = checkpoint.state.value in {"published", "failed", "canceled"}
    failed = checkpoint.state.value == "failed"
    canceled = checkpoint.state.value == "canceled"
    outcome: r.LifecycleOutcome = (
        "failed" if failed else "canceled" if canceled else "complete" if terminal else "running"
    )
    phase: r.LifecyclePhase = (
        "failed" if failed else "canceled" if canceled else "complete" if terminal else "rebuilding"
    )
    pending = 0 if terminal else max(0, checkpoint.expected_items - checkpoint.documents_built)
    return r.RebuildRunReport(
        generation_id=checkpoint.generation_id,
        state=checkpoint.state.value,
        expected_items=checkpoint.expected_items,
        next_sequence=checkpoint.next_sequence,
        documents_built=checkpoint.documents_built,
        chunks_built=checkpoint.chunks_built,
        vectors_reused=checkpoint.vectors_reused,
        vectors_embedded=checkpoint.vectors_embedded,
        diagnostic_code=(
            checkpoint.diagnostic_code.value if checkpoint.diagnostic_code is not None else None
        ),
        lifecycle=r.LifecycleProgress(
            phase=phase,
            outcome=outcome,
            acquired_items=checkpoint.documents_built,
            reused_items=checkpoint.vectors_reused,
            pending_items=pending,
            derived_generation_identity=checkpoint.generation_id,
            can_continue_offline=not terminal,
            estimated_remaining_items=pending,
        ),
    )


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

    confidence_reason: str = ""
    """Why the band is what it is, in the words retrieval used.

    Carried for the same reason the band is: the envelope has the score and nothing that makes
    a small score legible. ``search`` has said this since retrieval learned to admit ignorance,
    and ``ask`` showed the number alone — so the same underlying judgment was explained in one
    command and bare in the other."""

    message_id: str | None = None
    """Where the answer was persisted, when there was a conversation to persist it to."""

    expansions: tuple[r.GlossaryExpansion, ...] = ()
    """Glossary terms the question named, each with the passage its definition came from.

    Here rather than on the envelope for the same reason the band is: the envelope is what the
    generator produced, and this is a fact about the retrieval that produced its context. An
    answer reached through words the reader did not type has to be able to say so, and to name
    the document that put them there."""

    conflicts: tuple[r.GlossaryConflict, ...] = ()
    """Terms with more than one definition in scope, so none was expanded."""

    explicit_definition: bool = False
    """Whether the retrieval answered a question about a term by showing its definition.

    Here for the same reason the band is, and it is the *only* route it has: the generator's
    envelope carries the score and nothing else about the retrieval, so a classification left
    off this record would reach the non-streaming ``ask`` and the streamed ``final`` frame
    through no path at all. Both read it from here, which is what makes them the same answer.
    """

    payload: r.AnswerResultPayload | None = None
    """The settled result, built by :meth:`ApplicationService.ask_stream` when the run ends.

    Here rather than assembled by each caller, because there are now two consumers of the
    stream — :meth:`ApplicationService.ask` and the HTTP surface's SSE and websocket paths —
    and a payload rebuilt at the surface is a second answer to "what did this run produce".
    It is filled after ``message_id`` is known, so a streamed answer carries the id it was
    persisted under exactly as the non-streaming form does.
    """


def _awaiting_reparse(document: Document) -> bool:
    """Whether this document still serves the text it had before its identity was repaired.

    The migration records the ``content_hash`` it saw, for the documents whose parse changes and
    no others. While the column still holds that value the document has not been re-read; once it
    differs, it has. One comparison, and it needs no clock, no watermark and nothing that has to
    be cleaned up afterwards.
    """
    previous = document.metadata.get(PREVIOUS_IDENTITY)
    if not isinstance(previous, dict):
        return False
    recorded = previous.get("content_hash")
    return isinstance(recorded, str) and bool(recorded) and recorded == document.content_hash


def _sidecar_ordering_would_have_helped(moving: Sequence[Document]) -> bool:
    """Whether these rows are enriched pages that a conversion before the first sync would avoid.

    Narrow on purpose. The ordering advice is true for a corpus of enriched exports converted
    after it was indexed, and is *not* true for two files that genuinely declare one page id —
    converting those earlier changes nothing, because the conflict is in the documents rather
    than in when the manifests were written. Attaching it to both would be advice that does not
    apply to half the cases it appears in, which is how a diagnostic stops being read.

    **The signal is the outcome, not the presence of a record**, and reading the record alone was
    narrow in the docstring and wide in the code. Every adapted page carries one, whatever
    happened to it: ``invalid_metadata`` and ``ambiguous`` are recorded the same way, and for
    those "convert before the first sync" is advice about ordering offered for a problem that is
    not about ordering — the manifest was unreadable or the page declared an identity two files
    claim, and converting earlier reproduces both. Only ``identity_not_applied`` names a document
    that is path-keyed *because of when the conversion ran*, which is the one thing arriving
    earlier would have changed.

    Keyed off :attr:`~manicule.connectors.enriched.AdapterOutcome.IDENTITY_NOT_APPLIED` so that
    this and :func:`_identity_deliberately_unapplied` read the same field for the same reason. They
    are two halves of one judgment about that outcome, and having one test it while the other
    tested the record's mere existence is how they came to disagree.
    """
    return any(
        isinstance(record := document.metadata.get(ENRICHED_KEY), dict)
        and record.get("outcome") == AdapterOutcome.IDENTITY_NOT_APPLIED.value
        for document in moving
    )


def _identity_deliberately_unapplied(document: Document, *, claimed: set[str]) -> bool:
    """Whether this document is path-keyed on purpose rather than being one of two for one page.

    An enriched page with no manifest beside it is adapted, cited by its own page id, and keyed on
    its path **deliberately** — identity has to be known at discovery and discovery reads the
    manifest, not the document. It carries its own notice naming the page id and the command that
    applies it, so reporting it here as well would be one finding twice with two remedies, and the
    one this check gives is the wrong advice for it.

    **Unless the identity it declares is already held by another document**, and that exception is
    the whole reason this takes ``claimed``. Following the notice — generate the manifest, sync —
    creates the page-keyed document and leaves the path-keyed one behind, because the connector
    stops discovering it under the path it was stored by. Excluding it unconditionally meant the
    tool instructed an action that produces an orphan and then said nothing about it. Two rows for
    one page is exactly what this check is for, however they came to exist.

    Both halves were found by running a real ingest and reading what ``doctor`` said afterwards:
    first that "the manifest beside each one declares an identity" about a page with no manifest,
    then nothing at all about a corpus holding the same page twice.
    """
    record = document.metadata.get(ENRICHED_KEY)
    unapplied = (
        isinstance(record, dict)
        and record.get("outcome") == AdapterOutcome.IDENTITY_NOT_APPLIED.value
    )
    return unapplied and _declared(document) not in claimed


def _declared(document: Document) -> str:
    """The source identity this document's provenance record states, or ``""``."""
    record = document.provenance
    if record is None or record.source is None:
        return ""
    return record.source.source_id


_IDENTITY_SCAN = 10_000
"""How many documents the identity dry run examines before it stops and says it stopped.

A bound rather than a full scan, because ``doctor`` is run to read a sentence and a corpus of a
million rows should not be loaded to produce one. ``truncated`` is reported alongside the count so
that "no documents affected" cannot be read as "none anywhere" when it means "none in the first
ten thousand".

**Every state that reports a count reports its bound**, which is a rule rather than a preference.
The bound qualifies whatever the read concluded, so ``ok`` and ``degraded`` carry it on both
checks: the ok paths were the obvious gap, but ``document-content``'s *degraded* count was
bounded too and said so nowhere, and "3 documents are stale" from a scan that stopped at ten
thousand is the same misreading with a number in front of it. A count that omits its own bound is
the failure this constant exists to prevent, and omitting it is easiest on the path that looks
like good news.

``unknown`` is the exception and is not one: the read itself raised, so there is no page of
documents and no count to qualify. Its ``facts`` carry the ``error_type`` and nothing else, which
is the whole of what that state knows. Reporting ``examined: 0`` there would be a measurement
nobody took."""

_IDENTITY_SAMPLE = 25
"""How many of the affected documents are listed individually.

The count is the finding; the list is what makes it checkable. Twenty-five is enough to recognize
a pattern — one directory, one exporter, one space — without turning a diagnostic into an export."""


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
        collections: Sequence[str] = (),
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
            collections=collections,
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
        collections: Sequence[str] = (),
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
        query = self._query(
            question,
            limit=limit or 8,
            profile=profile,
            sources=sources,
            collection_ids=sorted(await self._collection_scope(collections)),
        )
        retriever = await self._backend.retriever()
        retrieved = await retriever.retrieve(query)
        documents = await self._require_scoped_context(retrieved)
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
            record.confidence_reason = confidence.reason
        record.expansions, record.conflicts = await self._glossary_payloads(retrieved.expansion)
        # After the expansions, because the classification is only reportable alongside the
        # provenance it names and `_glossary_payloads` is what decides whether that survived.
        record.explicit_definition = cited_definition(confidence, record.expansions)
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
                    question, envelope, record, started, conversation_id, documents=documents
                )

    async def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
        profile: str | None = None,
        sources: Sequence[str] = (),
        media_types: Sequence[str] = (),
        collections: Sequence[str] = (),
    ) -> r.SearchResult:
        """Rank passages without asking a model anything.

        ``collections`` names collections to search within, and combines with ``sources`` and
        ``media_types`` the way :class:`~manicule.core.retrieval.Filter` says every field
        does — disjunction within a field, conjunction between them. So two collections union,
        and a collection together with a source keeps only what is in both.
        """
        started = time.monotonic()
        query = self._query(
            query_text,
            limit=limit,
            profile=profile,
            sources=sources,
            media_types=media_types,
            collection_ids=sorted(await self._collection_scope(collections)),
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
                provenance=source_reference(documents.get(candidate.chunk.document_id)),
                score=candidate.score,
                scores=dict(candidate.scores),
                text=candidate.chunk.text,
                token_count=candidate.chunk.token_count,
            )
            for candidate in candidates
        )
        confidence = retrieved.confidence
        expansions, conflicts = await self._glossary_payloads(retrieved.expansion)
        return r.SearchResult(
            query=query_text,
            profile=query.profile.value,
            count=len(hits),
            hits=hits,
            confidence=confidence.score if confidence else None,
            confidence_band=confidence.band.value if confidence else None,
            confidence_reason=confidence.reason if confidence else "",
            explicit_definition=cited_definition(confidence, expansions),
            expansions=expansions,
            conflicts=conflicts,
            expanded_query=retrieved.expansion.expanded if retrieved.expansion else "",
            route=retrieved.trace.route.value,
            cached=retrieved.trace.cached,
            truncated=retrieved.context.truncated,
            elapsed_ms=_millis(started),
            collections=tuple(collections),
        )

    async def _glossary_payloads(
        self, expansion: QueryExpansion | None
    ) -> tuple[tuple[r.GlossaryExpansion, ...], tuple[r.GlossaryConflict, ...]]:
        """Turn what the glossary said into payloads, with every source resolved.

        **Every entry's document is looked up through the scoped store**, and an entry whose
        document this workspace cannot see is dropped rather than reported with a blank source.
        The entries came from a workspace-scoped lookup already, so this can only ever fire on
        a store that leaked — which is exactly why it is here and not assumed away. It is the
        same second check ``_require_scoped_chunks`` performs on the passages, applied to the
        one other thing a search now puts on screen.

        Dropping rather than raising, because a definition that has become unreadable between
        the lookup and the render is a race rather than an attack, and failing the whole search
        over it would turn a soft delete into an outage. A leak, by contrast, cannot get past
        this: there is no branch that emits an expansion without a resolved document.
        """
        if expansion is None or not (expansion.matches or expansion.conflicts):
            return (), ()
        entries = [match.entry for match in expansion.matches]
        entries += [entry for conflict in expansion.conflicts for entry in conflict.entries]
        documents = await self._scoped_documents({entry.document_id for entry in entries})

        def payload(entry: GlossaryEntry, matched: str, reason: str) -> r.GlossaryExpansion | None:
            document = documents.get(entry.document_id)
            if document is None:
                return None
            return r.GlossaryExpansion(
                document_id=entry.document_id,
                chunk_id=entry.chunk_id,
                uri=document.uri,
                title=document.title,
                provenance=source_reference(document),
                acronym=entry.acronym,
                display=entry.display,
                expansion=entry.expansion,
                matched=matched,
                reason=reason,
                form=entry.form.value,
                detection_confidence=entry.confidence,
                location=entry.location,
            )

        expanded = tuple(
            built
            for built in (
                payload(match.entry, match.surface, match.reason.value)
                for match in expansion.matches
            )
            if built is not None
        )
        conflicts: list[r.GlossaryConflict] = []
        for conflict in expansion.conflicts:
            candidates = tuple(
                built
                for built in (payload(entry, conflict.surface, "") for entry in conflict.entries)
                if built is not None
            )
            # A conflict whose candidates no longer resolve to two readable documents is not a
            # conflict this workspace has. Reporting it with one candidate would be reporting a
            # disagreement nobody can look at.
            if len(candidates) >= _CONFLICTING:
                conflicts.append(
                    r.GlossaryConflict(
                        acronym=conflict.key, matched=conflict.surface, candidates=candidates
                    )
                )
        return expanded, tuple(conflicts)

    # --- ingest ---------------------------------------------------------------------------

    async def index_path(
        self,
        path: Path | str,
        *,
        source: str = DEFAULT_SOURCE,
        limit: int | None = None,
        force: bool = False,
        watching: Watching | None = None,
    ) -> r.IngestReport:
        """Index a file or a directory.

        Args:
            path: What to index. A directory is walked recursively.
            source: The source name documents are recorded under. Part of their identity, so
                changing it re-indexes rather than updates.
            limit: Stop after this many discovered documents.
            force: Re-parse documents whose content has not changed.
            watching: Called with one sentence per document reaching a terminal outcome, for a
                caller streaming progress to somebody. ``None`` when nobody is watching.

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
            await asyncio.to_thread(target.resolve),
            name=source,
            limit=limit,
            force=force,
            watching=watching,
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
        indexed = skipped = failed = discovered = expanded = unrecorded = 0
        incomplete_report: r.IngestReport | None = None
        for path in changed:
            report = await self.index_path(path, source=source)
            discovered += report.discovered
            indexed += report.ingested
            skipped += report.skipped
            failed += report.failed
            expanded += report.expanded
            unrecorded += report.unrecorded
            for status, count in report.by_status.items():
                counts[status] = counts.get(status, 0) + count
            if report.retry_required:
                incomplete_report = report
                break
        if incomplete_report is not None:
            return totals.model_copy(
                update={
                    "discovered": discovered,
                    "ingested": indexed,
                    "skipped": skipped,
                    "failed": failed,
                    "expanded": expanded,
                    "by_status": counts,
                    "error": incomplete_report.error,
                    "outcome": "incomplete",
                    "enumeration_completed": False,
                    "watermark_advanced": False,
                    "retry_required": True,
                    "unrecorded": unrecorded,
                    "incomplete_reason": incomplete_report.incomplete_reason,
                    "elapsed_ms": _millis(started),
                }
            )
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

    async def connector_sync(
        self,
        name: str,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
        acquire_only: bool = False,
    ) -> r.IngestReport:
        """Run one configured connector.

        ``watching`` is called with one sentence each time a document reaches a terminal
        outcome. It exists for the proxied command line, where a sync runs in the server and the
        person who asked for it is looking at a different process's terminal — without it a long
        sync is indistinguishable from a hung one. Nothing else passes it, including the
        scheduler, which has nobody to tell.

        Raises:
            UnknownEntityError: Configuration has no connector by that name.
            PolicyError: The source is configured ``enabled = false``. Refused rather than run,
                because the setting existed and was reported by ``connector list`` while nothing
                consulted it — so an operator who disabled a source, and checked, was told it was
                off by the same program that would then sync it. A disabled source that syncs is
                worse than no switch at all: the switch is what they trusted.
        """
        started = time.monotonic()
        configured = self.settings.connectors.get(name)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {name!r}. Configured: {known}"
            raise UnknownEntityError(msg)
        if not configured.enabled:
            msg = (
                f"connector {name!r} is configured `enabled = false`. Set it to true in "
                f"[connectors.{name}] to sync it, or use `manicule index <path>` for a one-off "
                f"that does not change the source's configuration."
            )
            raise PolicyError(msg)
        ingestion = await self._backend.ingestion()
        report = await ingestion.sync(
            name, limit=limit, watching=watching, acquire_only=acquire_only
        )
        return _ingest_payload(report, started)

    async def connector_login(
        self,
        name: str,
        *,
        cookies: str = "",
        forget: bool = False,
        browser: bool = False,
        browser_state: Path | str | None = None,
        manual_cookie: bool = False,
        browser_provider: str | None = None,
        timeout_seconds: float | None = None,
        allow_insecure_state: bool = False,
        provider: BrowserSessionProvider | None = None,
        store: SessionStore | None = None,
    ) -> r.ConnectorSignedIn:
        """Capture the browser session a Confluence source authenticates with, or forget it.

        Three ways in, one way through. ``--browser`` drives a browser the person signs in to,
        ``--browser-state`` imports a Playwright state file somebody else's tooling wrote, and the
        default asks for a pasted ``Cookie`` header. All three end at
        :func:`~manicule.connectors.sessions.capture_cookies`, which proves the cookies against
        the instance before anything is stored — so a session copied short, taken from the wrong
        tab, extracted from a state file belonging to another site, or collected from a browser
        the person had not finished signing in to is refused rather than persisted. Otherwise the
        first thing to use it would be the first page of the next sync.

        **manicule never sees the password on any of the three.** There is no parameter here that
        could carry one and no code path that would accept one. On the browser path the person
        types into a browser manicule opened, which is a weaker statement than the other two —
        :mod:`manicule.connectors.browser` says exactly how much weaker and what replaces it.

        **A failed login leaves a working credential alone.** Nothing is removed before the
        replacement is verified, so a timeout, a closed window or a dead session costs the attempt
        and not the session already stored.

        Args:
            name: The configured source. It must exist, be enabled, and be a Confluence source,
                which is the only type that authenticates this way.
            cookies: The ``Cookie`` header from a signed-in browser. The manual path.
            forget: Remove the stored session instead of capturing one.
            browser: Open a browser and wait for the person to sign in.
            browser_state: Import cookies from a Playwright ``storage_state`` document.
            timeout_seconds: How long the browser path waits. ``None`` uses the source's
                ``browser_timeout_seconds``.
            allow_insecure_state: Import a state file other users on this machine can read.
            provider: What opens the browser. ``None`` uses Playwright. A parameter so the flow
                can be exercised without one, on the same principle as ``capture``'s ``transport``.
            store: Where the captured session goes. ``None`` uses this process's own vault,
                which is right for a server running this on its own behalf and wrong for the
                command line — there the browser opens where the person is and the credential
                has to end up in the server, so the command line passes a store that hands it
                over the control socket. A parameter rather than a branch on "am I a server",
                because a session's destination is a fact the caller knows and this does not.

        Raises:
            UnknownEntityError: Configuration has no connector by that name.
            PolicyError: The source is configured ``enabled = false``.
            ConfigError: That connector is not a Confluence source, more than one way in was
                asked for, the material carried no usable cookies, or the instance would not
                confirm who the session belongs to.
        """
        from manicule.connectors.config import (  # noqa: PLC0415 - no HTTP stack at import
            CONNECTOR_NAME,
            ConfluenceConfig,
        )
        from manicule.connectors.sessions import (  # noqa: PLC0415
            capture,
            capture_cookies,
            default_store,
        )

        chosen = [
            label
            for label, asked in (
                ("--browser", browser),
                ("--browser-provider", browser_provider is not None),
                ("--browser-state", browser_state is not None),
                ("--manual-cookie", manual_cookie),
                ("--forget", forget),
            )
            if asked
        ]
        if len(chosen) > 1:
            msg = (
                f"{' and '.join(chosen)} were given together, and each is a different thing to "
                f"do with this source's credential. Pick one."
            )
            raise ConfigError(msg)

        # Settings directly rather than ``Runtime.connector(name)``, which every other
        # connector operation uses. That builds the connector, and building it resolves the
        # credential — which is the thing this operation exists to capture, and which by
        # definition is not there yet the first time somebody runs this.
        configured = self.settings.connectors.get(name)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {name!r}. Configured: {known}"
            raise UnknownEntityError(msg)
        if not configured.enabled:
            msg = (
                f"connector {name!r} is configured `enabled = false`, so a credential captured "
                f"for it would not be used by anything. Set it to true in [connectors.{name}] "
                f"first."
            )
            raise PolicyError(msg)
        if configured.type != CONNECTOR_NAME:
            msg = (
                f"{name!r} is a {configured.type!r} source, and a browser session is how the "
                f"{CONNECTOR_NAME!r} connector authenticates against an instance behind an "
                f"identity provider. Nothing else has a session to capture."
            )
            raise ConfigError(msg)
        config = ConfluenceConfig.model_validate(configured.options)
        keeper = store if store is not None else default_store()
        if forget:
            await keeper.forget(config.base_url)
            return r.ConnectorSignedIn(
                name=name,
                base_url=config.base_url,
                account="",
                captured_at="",
                expires_at="",
                stored_in=keeper.describe(),
                forgotten=True,
            )

        selected = self._selected_provider(
            browser=browser,
            browser_provider=browser_provider,
            browser_state=browser_state,
            manual_cookie=manual_cookie,
            injected=provider,
        )
        if selected is BrowserProvider.BROWSER_STATE:
            if browser_state is None:
                msg = (
                    f"the {selected.value!r} provider imports a Playwright storage_state "
                    f"document and none was named. Pass `--browser-state <file>`, or choose a "
                    f"provider that does not need one."
                )
                raise ConfigError(msg)
            session = await capture_cookies(
                config,
                _state_cookies(config, browser_state, allow_insecure=allow_insecure_state),
                store=keeper,
            )
        elif selected is BrowserProvider.MANUAL_COOKIE:
            session = await capture(config, cookies, store=keeper)
        else:
            session = await capture_cookies(
                config,
                await self._browser_cookies(
                    config,
                    provider=provider,
                    timeout_seconds=timeout_seconds,
                    selected=selected,
                ),
                store=keeper,
            )
        expires = session.captured_at + timedelta(hours=config.session_max_age_hours)
        return r.ConnectorSignedIn(
            name=name,
            base_url=config.base_url,
            account=session.account,
            captured_at=session.captured_at.isoformat(),
            expires_at=expires.isoformat(),
            stored_in=keeper.describe(),
        )

    def _selected_provider(
        self,
        *,
        browser: bool,
        browser_provider: str | None,
        browser_state: Path | str | None,
        manual_cookie: bool,
        injected: BrowserSessionProvider | None,
    ) -> BrowserProvider:
        """This workspace's answer to :func:`selected_provider`."""
        return selected_provider(
            browser=browser,
            browser_provider=browser_provider,
            browser_state=browser_state,
            manual_cookie=manual_cookie,
            injected=injected,
            configured=self.settings.authentication.confluence.default_provider,
        )

    async def _browser_cookies(
        self,
        config: ConfluenceConfig,
        *,
        provider: BrowserSessionProvider | None,
        timeout_seconds: float | None,
        selected: BrowserProvider = BrowserProvider.BUNDLED_CHROMIUM,
    ) -> Mapping[str, SecretStr]:
        """Sign in through a browser and return the cookies that belong to this instance.

        The filter is applied *here* rather than inside the provider, so that every provider —
        Playwright today, something else later — is held to it by the caller rather than by its
        own good behavior. An identity provider's cookies never reach the store because they
        never get past this line.
        """
        from manicule.connectors.browser import (  # noqa: PLC0415 - optional dependency
            origin_cookies,
        )

        driver = provider if provider is not None else self._driver_for(selected)
        # `is None` rather than `or`: a `--timeout 0` is falsy, and folding it into the default
        # would silently wait five minutes for somebody who asked to wait none. Zero is refused
        # at the command line rather than substituted, so neither reading is invented here.
        wait = config.browser_timeout_seconds if timeout_seconds is None else timeout_seconds
        candidates = await driver.authenticate(config, timeout_seconds=wait)
        found = origin_cookies(candidates, base_url=config.base_url)
        if not found:
            msg = (
                f"the browser finished with no cookies for {config.base_url}. Nothing has been "
                f"stored and any previously stored session is untouched. If sign-in redirected "
                f"to a different host than the one configured, check base_url names the site "
                f"root including any context path."
            )
            raise ConfigError(msg)
        return found

    def _driver_for(self, selected: BrowserProvider) -> BrowserSessionProvider:
        """Build the provider object `selected` names, or refuse without trying another.

        **Every failure here is a refusal rather than a substitution**, which is the whole of
        acceptance criterion 4. A browser that is not installed, a name that matches two of
        them, a profile directory that cannot be made safely — each raises
        `ProviderRefusedError` naming what this machine does have, and none of them returns a
        different provider that happens to work.

        Raises:
            ProviderRefusedError: The named provider cannot be built on this machine.
        """
        from manicule.connectors.browser import (  # noqa: PLC0415 - optional dependency
            InstalledChromiumProvider,
            PlaywrightProvider,
            private_profile,
            resolve_browser,
        )

        if selected is not BrowserProvider.INSTALLED_CHROMIUM:
            return PlaywrightProvider()
        auth = self.settings.authentication.confluence
        profile = auth.profile_dir or (self.settings.data_dir / "browser-auth" / "default")
        return InstalledChromiumProvider(
            executable=resolve_browser(auth.installed_browser),
            profile_dir=private_profile(profile),
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
            recorded = metadata.get("last_run")
            last_run = cast("dict[str, Any]", recorded) if isinstance(recorded, dict) else {}
            raw_outcome = last_run.get("outcome")
            last_outcome: r.IngestOutcome | None = (
                cast("r.IngestOutcome", raw_outcome)
                if raw_outcome in {"complete", "bounded", "incomplete"}
                else None
            )
            raw_enumeration_completed = last_run.get("enumeration_completed")
            raw_watermark_advanced = last_run.get("watermark_advanced")
            raw_lifecycle = last_run.get("lifecycle")
            last_lifecycle: r.LifecycleProgress | None = None
            if isinstance(raw_lifecycle, dict):
                # Connector metadata is diagnostic history and may predate this contract. One
                # malformed old status must not hide the configured connector itself.
                with contextlib.suppress(ValidationError):
                    last_lifecycle = r.LifecycleProgress.model_validate(raw_lifecycle)
            summaries.append(
                r.ConnectorSummary(
                    name=name,
                    type=configured.type,
                    enabled=configured.enabled,
                    installed=configured.type in installed if discovery else True,
                    last_synced_at=_text_or_none(metadata.get("last_synced_at")),
                    status=_text(metadata.get("status")),
                    documents=await store.count_documents(source=name),
                    last_outcome=last_outcome,
                    retry_required=last_run.get("retry_required") is True,
                    last_error_type=_text(last_run.get("error_type")),
                    last_enumeration_completed=(
                        raw_enumeration_completed
                        if isinstance(raw_enumeration_completed, bool)
                        else None
                    ),
                    last_watermark_advanced=(
                        raw_watermark_advanced if isinstance(raw_watermark_advanced, bool) else None
                    ),
                    last_lifecycle=last_lifecycle,
                    full_inventory_authority=_configured_full_inventory_authority(configured),
                )
            )
        return r.ConnectorList(count=len(summaries), connectors=tuple(summaries))

    async def snapshot_status(self, connector: str) -> r.SnapshotStatusReport:
        """Read aggregate state for the active durable snapshot of one configured source."""
        if connector not in self.settings.connectors:
            raise UnknownEntityError(f"no connector named {connector!r}")
        ingestion = await self._backend.ingestion()
        found = await ingestion.snapshot_status(connector)
        if found is None:
            raise UnknownEntityError("this connector has no durable source snapshot")
        run, verified = found
        return _snapshot_status_report(run, verified, verification_performed=False)

    async def snapshot_verify(self, run_id: str) -> r.SnapshotStatusReport:
        """Verify one workspace-owned manifest without exposing any member identity."""
        ingestion = await self._backend.ingestion()
        found = await ingestion.snapshot_verify(run_id)
        if found is None:
            raise UnknownEntityError("no durable source snapshot has that id")
        run, verified = found
        return _snapshot_status_report(run, verified, verification_performed=True)

    async def rebuild_plan(self, snapshot_id: str) -> r.RebuildPlanReport:
        """Price a connector-free replacement generation from retained source bytes."""
        ingestion = await self._backend.ingestion()
        try:
            estimate = await ingestion.rebuild_plan(snapshot_id)
        except RebuildOperationError:
            raise
        except RebuildTerminalGenerationError as exc:
            raise RebuildTerminalError from exc
        except RebuildLeaseConflictError as exc:
            raise RebuildLeaseError from exc
        except RebuildPublicationValidationError as exc:
            raise RebuildValidationError from exc
        except (SQLAlchemyError, OSError) as exc:
            raise RebuildStorageError from exc
        return _rebuild_plan_report(estimate)

    async def rebuild_run(self, snapshot_id: str) -> r.RebuildRunReport:
        """Execute or resume the deterministic generation for one snapshot and target."""
        ingestion = await self._backend.ingestion()
        try:
            checkpoint = await ingestion.rebuild_run(snapshot_id, secrets.token_urlsafe(24))
        except RebuildOperationError:
            raise
        except RebuildTerminalGenerationError as exc:
            raise RebuildTerminalError from exc
        except RebuildLeaseConflictError as exc:
            raise RebuildLeaseError from exc
        except RebuildPublicationValidationError as exc:
            raise RebuildValidationError from exc
        except (SQLAlchemyError, OSError) as exc:
            raise RebuildStorageError from exc
        return _rebuild_run_report(checkpoint)

    async def rebuild_status(self, generation_id: str) -> r.RebuildRunReport:
        """Read one workspace-owned offline rebuild checkpoint."""
        ingestion = await self._backend.ingestion()
        try:
            checkpoint = await ingestion.rebuild_status(generation_id)
        except RebuildOperationError:
            raise
        except RebuildTerminalGenerationError as exc:
            raise RebuildTerminalError from exc
        except RebuildLeaseConflictError as exc:
            raise RebuildLeaseError from exc
        except RebuildPublicationValidationError as exc:
            raise RebuildValidationError from exc
        except (SQLAlchemyError, OSError) as exc:
            raise RebuildStorageError from exc
        if checkpoint is None:
            raise UnknownEntityError("no durable offline rebuild has that generation id")
        return _rebuild_run_report(checkpoint)

    async def connector_sidecar(
        self,
        root: Path | str | None = None,
        *,
        source: str = "",
        force: bool = False,
    ) -> r.SidecarReport:
        """Write a sidecar manifest beside every enriched HTML page under a root.

        Enriched standalone HTML — one file per page, carrying the page's own id, space, version,
        modification time and canonical address in a machine-identifiable section — is indexed
        perfectly well by ordinary filesystem ingestion today, and cited by its filename and a
        ``file://`` URI, because that is all the connector was given. This reads the identity out
        of each page and records it in the manifest format the filesystem connector *already*
        reads, so the pages become properly citable without a new connector, a new ingestion path
        or a change to the provenance interface.

        **The source files are never modified.** The manifest is written beside each page, at a
        path derived from where the walk found the file, so nothing read out of a document can
        influence where anything is written.

        **``source`` is what makes a custom exporter convertible at all.** Without it the run uses
        the built-in default profile, which is correct for a corpus that matches the default and
        is *predictably useless* for one that does not: every page reports ``no_profile`` and no
        manifest is written, while the configured sync over the same directory adapts all of them.
        Naming a configured source resolves the root and the profiles from the connector the sync
        would run — :meth:`~manicule.app.ports.Ingesting.connector`, the same object, from the
        same container — so the two cannot hold different readings of one profile.

        Args:
            root: The directory of pages. Walked without following symlinks. With ``source``
                it is optional and bounded: omitted it means the connector's whole root, and
                supplied it must resolve **inside** that root, so a configured source cannot be
                used as a handle for converting an unrelated directory.
            source: A configured connector *instance* name — the key in ``[connectors.<name>]``,
                never a connector type. It must exist, be enabled, be a filesystem source, and
                declare at least one enriched profile.
            force: Replace manifests that already exist. Off by default, because one already
                there was most likely written by hand or by another tool.

        Raises:
            UnknownEntityError: ``root`` is not a directory, or no connector is configured
                under ``source``.
            PolicyError: ``source`` is configured ``enabled = false``. The same refusal
                :meth:`connector_sync` gives, for the same reason: a switch an operator set and
                checked has to mean something on every surface that reads it, and converting a
                disabled source's corpus is preparing an ingest that will not run.
            ConfigError: ``source`` is not a filesystem connector, declares no enriched profile,
                or was named alongside a root outside its own tree. Also raised when neither a
                root nor a source is given, because there is then no directory to walk.
        """
        from manicule.connectors.enriched import DEFAULT_PROFILE  # noqa: PLC0415
        from manicule.connectors.enriched_html import write_sidecars  # noqa: PLC0415

        profiles: tuple[EnrichedProfile, ...] = (DEFAULT_PROFILE,)
        if source:
            connector = await self._filesystem_source(source)
            profiles = connector.profiles
            path = await asyncio.to_thread(_bounded_root, root, connector.root, source)
        elif root is not None:
            path = _local(root)
        else:
            msg = (
                "name a directory to convert, or a configured source with `--source <name>`. "
                "`manicule connector list` names the configured ones."
            )
            raise ConfigError(msg)
        if not await asyncio.to_thread(path.is_dir):
            msg = f"no such directory: {path}"
            raise UnknownEntityError(msg)
        root_path = path.resolve()
        # `profiles` exactly as configured, in configured order, with nothing appended. Adding
        # the default to a connector that omitted it would make this run adapt pages the sync
        # will not, which is the disagreement the whole `--source` path exists to remove.
        outcomes = await asyncio.to_thread(write_sidecars, path, force=force, profiles=profiles)
        counts: dict[str, int] = {}
        for outcome in outcomes:
            counts[outcome.outcome.value] = counts.get(outcome.outcome.value, 0) + 1
        return r.SidecarReport(
            root=str(root_path),
            source=source,
            profiles=tuple(profile.name for profile in profiles),
            considered=len(outcomes),
            written=sum(1 for outcome in outcomes if outcome.written),
            # Every considered file lands in exactly one bucket, adapted ones included, so the
            # counts sum to `considered` and a run that adapted nothing states which kind of
            # nothing it was rather than leaving that to be inferred from an empty `written`.
            by_outcome=counts,
            skipped=tuple(
                # Relative to the root, which the report has already named. Absolute paths here
                # repeat the root on every row and push the reason — the part that says what to
                # do next — off the side of a terminal table.
                r.SidecarSkip(
                    path=_relative_to(outcome.path, root_path),
                    reason=outcome.skipped_reason,
                    outcome=outcome.outcome.value,
                )
                for outcome in outcomes
                if not outcome.written
            ),
        )

    async def _filesystem_source(self, name: str) -> FilesystemConnector:
        """The constructed filesystem connector for a configured source, or a stated refusal.

        Every check here answers "would naming this source produce a conversion that agrees with
        the sync?", and each one fails loudly rather than degrading to the default profile —
        which is the behavior being removed. A silent fallback would put the operator back where
        they started: a command that runs, reports ``no_profile`` for every page, and leaves them
        to work out that the flag they passed did nothing.

        Ordered so that the cheap configuration questions are answered before anything is built.
        A disabled or mistyped source should not construct a connector to be told it was the
        wrong one.
        """
        from manicule.connectors.config import FILESYSTEM_CONNECTOR_NAME  # noqa: PLC0415
        from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415

        configured = self.settings.connectors.get(name)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            # Named as an instance rather than a type, because the commonest mistake is passing
            # `filesystem` — the connector's *kind* — and getting an error that does not say
            # which of the two kinds of name was wanted.
            msg = (
                f"no connector named {name!r}. `--source` names a configured instance — the key "
                f"in [connectors.<name>] — rather than a connector type. Configured: {known}"
            )
            raise UnknownEntityError(msg)
        if not configured.enabled:
            msg = (
                f"connector {name!r} is configured `enabled = false`. Set it to true in "
                f"[connectors.{name}] to convert its corpus, or pass the directory as an "
                f"argument for a one-off that does not consult the source's configuration."
            )
            raise PolicyError(msg)
        if configured.type != FILESYSTEM_CONNECTOR_NAME:
            msg = (
                f"{name!r} is a {configured.type!r} source, and sidecar generation writes into a "
                f"local directory of exported pages. Only a {FILESYSTEM_CONNECTOR_NAME!r} source "
                f"has one."
            )
            raise ConfigError(msg)
        ingestion = await self._backend.ingestion()
        connector = await ingestion.connector(name)
        if not isinstance(connector, FilesystemConnector):
            # Reachable only where a plugin has registered something else under the filesystem
            # name. Refused rather than duck-typed: the two attributes read below decide where
            # this writes and what it recognizes, and guessing that an unknown class means the
            # same thing by them is how a conversion ends up rooted somewhere nobody named.
            # "builds an instance of X" rather than "builds a X": the type name is interpolated,
            # so no fixed article is right for all of them — the class this actually fires on
            # most often is `object`, and the message read "builds a object".
            msg = (
                f"{name!r} is configured as a {FILESYSTEM_CONNECTOR_NAME!r} source but builds an "
                f"instance of {type(connector).__name__}, which does not declare a root and "
                f"enriched profiles this can convert with."
            )
            raise ConfigError(msg)
        if not connector.profiles:
            msg = (
                f"{name!r} declares an empty `enriched_profiles`, which turns adaptation off: "
                f"every HTML file under it is indexed as ordinary HTML and none is an enriched "
                f"export. A conversion under that configuration would report `no_profile` for "
                f"every page and write nothing. Configure the profile this exporter uses, or "
                f"pass the directory as an argument to convert it with the built-in default."
            )
            raise ConfigError(msg)
        return connector

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
        detail = "; ".join([*report.unrepairable, *report.failures, *report.superseded])
        # **``superseded`` is read first, and both halves of that matter.** Ahead of
        # ``unchanged``, because a refused re-parse is not an unchanged one: nothing was
        # compared and the commit was declined, and ``unchanged`` with an empty detail is the
        # one answer that tells an operator to stop looking. Ahead of ``failed``, because
        # nothing failed — the corpus holds newer text than this run was working from, which is
        # the outcome anybody would have wanted.
        if report.superseded:
            status = "superseded"
        elif report.documents:
            status = "reindexed"
        elif detail:
            status = "failed"
        else:
            status = "unchanged"
        return r.DocumentReindexed(
            document_id=document_id, status=status, chunks=report.chunks, detail=detail
        )

    async def reembed_plan(self) -> r.ReembedPlanReport:
        """Price the exact configured embedder migration without embedding or source access."""
        ingestion = await self._backend.ingestion()
        plan, target_identity, target_dimension = await ingestion.reembed_plan()
        return r.ReembedPlanReport(
            documents=plan.documents,
            chunks=plan.chunks,
            input_bytes=plan.input_bytes,
            estimated_seconds=plan.estimated_seconds,
            peak_memory_bytes=plan.peak_memory_bytes,
            temporary_disk_bytes=plan.temporary_disk_bytes,
            unrepairable_documents=plan.unrepairable_documents,
            target_identity=target_identity,
            target_dimension=target_dimension,
            lifecycle=r.LifecycleProgress(
                phase="complete",
                outcome="complete",
                dry_run=True,
                enumerated_items=plan.chunks,
                failed_items=plan.unrepairable_documents,
                derived_generation_identity=target_identity,
                can_continue_offline=True,
                estimated_remaining_items=plan.chunks,
                estimated_remaining_seconds=plan.estimated_seconds,
            ),
        )

    async def reembed_start(self, run_id: str) -> r.ReembedRunReport:
        """Create a durable plan and return its recovery id before embedding begins."""
        ingestion = await self._backend.ingestion()
        run = await ingestion.reembed_start(run_id, secrets.token_urlsafe(24))
        return _reembed_run_report(run)

    async def reembed_resume(self, run_id: str) -> r.ReembedRunReport:
        """Resume one durable migration after a process exit or transient refusal."""
        ingestion = await self._backend.ingestion()
        return _reembed_run_report(
            await ingestion.reembed_resume(run_id, secrets.token_urlsafe(24))
        )

    async def reembed_status(self, run_id: str) -> r.ReembedRunReport:
        """Read aggregate durable progress without exposing snapshot or source identities."""
        ingestion = await self._backend.ingestion()
        run = await ingestion.reembed_status(run_id)
        if run is None:
            raise UnknownEntityError("no durable re-embedding run has that id")
        return _reembed_run_report(run)

    async def reembed_abandon(self, run_id: str) -> r.ReembedRunReport:
        """Mark one unfinished run failed so its unpublished generation can be cleaned."""
        ingestion = await self._backend.ingestion()
        return _reembed_run_report(
            await ingestion.reembed_abandon(run_id, secrets.token_urlsafe(24))
        )

    async def reembed_cleanup(self, run_id: str) -> r.ReembedCleanupReport:
        """Remove only terminal, non-live shadow storage for one durable run."""
        ingestion = await self._backend.ingestion()
        return r.ReembedCleanupReport(
            run_id=run_id,
            removed=await ingestion.reembed_cleanup(run_id),
        )

    async def reembed_recover_pending(self) -> ReembedRecovery:
        """Internal serving-process restart recovery; each run remains inspectable normally."""
        ingestion = await self._backend.ingestion()
        return await ingestion.reembed_recover_pending()

    async def document_reindex_stale(
        self, *, batch: int = DEFAULT_SWEEP_BATCH, dry_run: bool = False
    ) -> r.StaleReparseReport:
        """Re-parse every document a parser upgrade left behind. Touches no network.

        The corpus-wide end of :meth:`document_reindex`. A parser's fingerprint records the
        version that produced a document's text, so bumping one makes every document it read
        stale at once — and until this ran, closing that window meant waiting for each document
        to come round again on a connector sync, with every anchor stored under the old version
        being resolved by the new one in the meantime.

        Nothing here is fetched. Every byte comes out of the blob store, and a document whose
        bytes were never retained is named with the reason rather than failing the sweep.

        Args:
            batch: Documents per page of the selection.
            dry_run: Report the selection and write nothing at all.
        """
        ingestion = await self._backend.ingestion()
        sweep = await ingestion.reparse_stale(batch=batch, dry_run=dry_run)
        return r.StaleReparseReport(
            dry_run=sweep.dry_run,
            selected=sweep.selected,
            reparsed=sweep.reparsed,
            unchanged=sweep.unchanged,
            changed=sweep.changed,
            chunks_new=sweep.chunks_new,
            chunks_kept=sweep.chunks_kept,
            embedding=r.EmbeddingCost(
                reused=sweep.embedding.reused,
                embedded=sweep.embedding.embedded,
                input_changed=sweep.embedding.input_changed,
                first_seen=sweep.embedding.first_seen,
                repaired=sweep.embedding.repaired,
                forward_calls=sweep.embedding.forward_calls,
                cache_hits=sweep.embedding.cache_hits,
                vectors_new=sweep.embedding.vectors_new,
                vectors_replaced=sweep.embedding.vectors_replaced,
                vectors_backfilled=sweep.embedding.vectors_backfilled,
            ),
            unrepairable=sweep.unrepairable,
            failed=sweep.failed,
            unrepairable_documents=tuple(sweep.unrepairable_documents),
            failures=tuple(sweep.failures),
            superseded=sweep.superseded,
            superseded_documents=tuple(sweep.superseded_documents),
        )

    async def document_redetect_glossary(
        self, *, batch: int = DEFAULT_SWEEP_BATCH, dry_run: bool = False
    ) -> r.StaleGlossaryReport:
        """Bring every document's glossary on to the installed detector's rules.

        **The repair a detector change needs, and the one no other verb performs.** Detection
        runs at ingest, after parsing; a re-sync of unchanged bytes skips before it; and
        ``parse_fp`` does not move when a detection rule does. So a corpus indexed before a
        detector fix keeps the entries the old rules produced, reports current parser, chunker
        and embedder fingerprints, and is right about all three.

        Nothing here parses, fetches or embeds. It reads stored chunk text and writes rows, so
        the cost is a pass over the corpus's text and no GPU time at all — which is why it is
        its own command rather than a wider ``--stale`` sweep that would charge a re-parse and a
        re-embed for a change to a regular expression.

        Args:
            batch: Documents per page of the selection.
            dry_run: Report the selection and write nothing at all.

        Raises:
            PolicyError: ``rag.glossary.detect_on_ingest`` is off, so there is no detector for
                the corpus to be current with.
        """
        ingestion = await self._backend.ingestion()
        sweep = await ingestion.redetect_stale_glossary(batch=batch, dry_run=dry_run)
        return r.StaleGlossaryReport(
            dry_run=sweep.dry_run,
            selected=sweep.selected,
            redetected=sweep.redetected,
            unchanged=sweep.unchanged,
            changed=sweep.changed,
            entries_before=sweep.entries_before,
            entries_after=sweep.entries_after,
            failed=sweep.failed,
            failures=tuple(sweep.failures),
            unrepairable=sweep.unrepairable,
            unrepairable_documents=tuple(sweep.unrepairable_documents),
            superseded=sweep.superseded,
            superseded_documents=tuple(sweep.superseded_documents),
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
        detector = await (await self._backend.ingestion()).glossary_fingerprint()
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
            weights_ref=fingerprints.embed.weights_ref if fingerprints.embed else None,
            weights_identity=fingerprints.embed.weights_identity if fingerprints.embed else None,
            chunk_fingerprint=fingerprints.chunk.canonical() if fingerprints.chunk else None,
            # The fourth stage, reported beside the other three. There is no corpus-wide
            # committed value to read out of `index_state` for this one — detection is compared
            # per document, like parsing — so what is shown is what the *installed* detector
            # would produce, and the count beside it is how much of the corpus disagrees.
            glossary=detector.describe(),
            stale_glossary=await store.count_documents(glossary_fp_other_than=detector.canonical()),
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

    async def doctor(self, *, fix: bool = False) -> r.Diagnosis:
        """Check what can be checked without building anything expensive.

        Every check answers one question and says what to do when the answer is bad. It does
        not load a model runtime, does not open a provider connection and does not read a
        document — a diagnostic that needs the system working is not a diagnostic.

        Args:
            fix: Perform the repairs this command knows how to perform, and then report the
                state that resulted. Two repairs exist, and they are the same repair against
                two libraries that ship the same gap: seeding the declared code grammars
                (:meth:`_grammar_check`) and the BPE vocabularies the token counters measure
                with (:meth:`_vocabulary_check`). They are the only things here that write to
                the machine or may use the network.

                **Off by default, and passed only by the command line.** The MCP tool and the
                HTTP route both call this with no arguments, deliberately: a diagnostic an
                assistant can reach should not be able to start a download, and a repair is a
                thing an operator asks for rather than something a health page does on their
                behalf. ``tests/app/test_surface_parity.py`` holds that line.
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
        checks.append(await self._glossary_check())
        checks.append(await self._connectors_check())
        checks.append(await self._sessions_check())
        checks.append(await self._document_identity_check())
        checks.append(await self._document_content_check())
        checks.append(await self._wiki_provenance_check())
        checks.append(await self._grammar_check(fix=fix))
        checks.append(await self._vocabulary_check(fix=fix))
        # No `fix`. The other two repairs move megabytes and finish while somebody is looking
        # at the command; this one would move over a gigabyte, and `doctor --fix` is what a
        # person runs to un-break a machine, not what they run to spend ten minutes. It
        # reports, and names the command that fetches.
        checks.append(await self._model_check())
        checks.extend(await self._backend.component_checks())
        worst: r.CheckState = "ok"
        for check in checks:
            if _severity(check.state) > _severity(worst):
                worst = check.state
        return r.Diagnosis(state=worst, checks=tuple(checks))

    def _configuration_check(self) -> r.Check:
        problems = self.settings.policy_problems()
        if not problems:
            return r.Check(
                name="configuration",
                state="ok",
                detail="no policy conflicts",
                facts={"problems": []},
            )
        return r.Check(
            name="configuration",
            state="failing",
            detail="; ".join(problems),
            facts={"problems": list(problems)},
            remedy="manicule config show",
        )

    def _transport_check(self) -> r.Check:
        transport = self.settings.security.transport
        mode = self.settings.security.auth.mode
        # The bind host and the authentication mode, and neither is a secret: a host is an
        # address this machine already answers on, and the mode is which *kind* of credential
        # is demanded rather than any credential itself.
        facts: dict[str, JsonValue] = {
            "bind_host": transport.bind_host,
            "loopback": transport.is_loopback,
            "auth_mode": mode.value,
        }
        if transport.is_loopback:
            return r.Check(
                name="transport",
                state="ok",
                detail=f"bound to {transport.bind_host}, reachable only from this machine",
                facts=facts,
            )
        if mode is AuthMode.NONE:
            return r.Check(
                name="transport",
                state="failing",
                detail=(
                    f"security.transport.bind_host is {transport.bind_host!r} with no "
                    f"authentication. Bind 127.0.0.1, or set security.auth.mode."
                ),
                facts=facts,
                remedy="manicule config set security.transport.bind_host 127.0.0.1",
            )
        return r.Check(
            name="transport",
            state="degraded",
            detail=(
                f"bound to {transport.bind_host}, which is reachable from the network. "
                f"Authentication is on ({mode.value})."
            ),
            facts=facts,
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
            facts={
                "plugins": len(discovery.manifests),
                "components": len(discovery.registry),
                "disabled": len(discovery.disabled),
            },
        )

    async def _storage_check(self) -> r.Check:
        """Whether the database is at a schema revision, and what it means when it is not.

        The remedy used to be "run ``manicule init``", and that was a repair nobody had
        written: ``init`` chooses an embedding backend and writes a configuration file, and it
        does not open the database at all. Migrations run when the document store is first
        built — which is what asking this question just did, one line above. So an answer of
        "no revision" is not a step somebody skipped; it is migrations that ran and did not
        stick, and the fixes for that are about the directory rather than about a command.
        """
        try:
            maintenance = await self._backend.maintenance()
            revision = await maintenance.schema_revision()
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="storage",
                state="failing",
                detail=f"{type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        where = r.redacted_path(self.settings.data_dir)
        if revision is None:
            return r.Check(
                name="storage",
                state="degraded",
                detail=(
                    f"the database in {where} carries no schema revision. "
                    f"Opening it applies them, and opening it is what this check just did, so "
                    f"they did not take: check the directory is writable by the account "
                    f"running manicule and that the database file is not from a later version."
                ),
                facts={"data_dir": where, "revision": None},
            )
        return r.Check(
            name="storage",
            state="ok",
            detail=f"schema at {revision}",
            facts={"data_dir": where, "revision": revision},
        )

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
        # Redacted, not withheld. The mode and the location are the whole diagnosis, and
        # `chmod 0700 ~/…` is still a command the operator who owns that home can paste.
        where = r.redacted_path(data_dir)
        if os.name != "posix":
            return r.Check(
                name="permissions",
                state="unknown",
                detail=f"{where} is on a platform with no POSIX file modes to check",
                facts={"path": where, "posix": False},
            )
        try:
            exposed = await asyncio.to_thread(exposure, data_dir)
        except OSError as exc:
            # "Cannot be examined" is not "is exposed", and reporting it as the latter would
            # send an operator to chmod a path that may not be there.
            return r.Check(
                name="permissions",
                state="unknown",
                detail=f"{where} could not be examined: {exc}",
                facts={"path": where, "error_type": type(exc).__name__},
            )
        if not exposed:
            return r.Check(
                name="permissions",
                state="ok",
                detail=f"{where} is readable only by the account running manicule",
                facts={"path": where, "exposed": False},
            )
        return r.Check(
            name="permissions",
            state="failing",
            detail=(
                f"{where} carries group or other permissions ({exposed:03o}), so it is "
                f"reachable by accounts other than the one running manicule. It holds the "
                f"retained source bytes of every indexed document, which makes this an "
                f"exposure of the corpus rather than a tidiness problem. Run "
                f"`chmod 0700 {where}`."
            ),
            # `exposed_bits` rather than `mode`: :func:`exposure` returns the group and other
            # bits alone, not the directory's mode. A field named `mode` carrying `055` would
            # be read as the whole mode by everyone who did not go and check.
            facts={"path": where, "exposed": True, "exposed_bits": f"{exposed:03o}"},
            remedy=f"chmod 0700 {where}",
        )

    async def _connectors_check(self) -> r.Check:
        """Documents filed under a connector *type* while an instance of that type is configured.

        Before #94, ``Connector.name`` was the component name, so ``[connectors.team-handbook]``
        stored ``source = "confluence-snapshot"``. It is now the instance name — which changes
        what ``document_id(workspace_id, source, source_id)`` derives from, for exactly those
        corpora. This is the check that tells an operator before their next sync does.

        **``degraded``, not ``failing``, and the distinction is deliberate.** Nothing is broken:
        the documents are indexed, searchable and correctly cited, and every query over them
        works. What is true is that their identity is about to change, which is a thing to act on
        rather than a fault to repair — the same reading the vocabulary check applies to a
        capability that is absent rather than damaged. ``failing`` would be defensible if a
        silent overwrite had *already* happened, but that requires two instances of one type to
        have both synced, which the pre-#94 code made impossible: they shared one global root and
        therefore one document set. So the damage this warns about is prospective, and
        ``degraded`` says so.

        **The remedy named here is one that has been run.** There is deliberately no mention of
        reconciliation: :func:`manicule.ingest.reconcile.reconcile` exists and is tested, but
        nothing in the product calls it and no command invokes it, so naming it would send an
        operator looking for a regime that is not reachable.
        """
        configured = self.settings.connectors
        if not configured:
            return r.Check(
                name="connectors",
                state="ok",
                detail="no sources configured",
                facts={"affected": []},
            )
        # Only an instance whose name differs from its type moves. `[connectors.filesystem]
        # type = "filesystem"` stored `source = "filesystem"` before and stores it now, so its
        # documents keep their ids and reporting it would be a warning about nothing.
        renamed = {
            settings.type: name
            for name, settings in sorted(configured.items())
            if name != settings.type
        }
        if not renamed:
            return r.Check(
                name="connectors",
                state="ok",
                detail="every configured source is named after its own type, so no document "
                "identity depends on the change in #94",
                facts={"affected": []},
            )
        try:
            store = await self._backend.documents()
            stale = {
                type_name: count
                for type_name in renamed
                if (count := await store.count_documents(source=type_name))
            }
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="connectors",
                state="unknown",
                detail=f"the corpus could not be examined: {type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        if not stale:
            return r.Check(
                name="connectors",
                state="ok",
                detail="no documents are filed under a connector type name",
                facts={"affected": []},
            )
        listed = ", ".join(
            f"{name!r} ({count} document(s))" for name, count in sorted(stale.items())
        )
        instances = ", ".join(f"{renamed[name]!r}" for name in sorted(stale))
        first = sorted(stale)[0]
        return r.Check(
            name="connectors",
            state="degraded",
            detail=(
                f"{listed} are filed under a connector *type* name, while the source(s) "
                f"configured for those types are named {instances}. manicule now records a "
                f"document's source as the configured instance, so the next sync will index "
                f"these pages again under new ids and leave the existing rows behind — the "
                f"corpus will appear to have doubled, and nothing will report an error. "
                f"Inspect them with `manicule document list --source {first}`, and remove each "
                f"with `manicule document delete <id>` before or after re-syncing. There is no "
                f"bulk reconciliation for this and none is implied."
            ),
            facts=cast(
                "dict[str, JsonValue]",
                {
                    "affected": sorted(stale),
                    "documents": sum(stale.values()),
                    "instances": sorted(renamed[name] for name in stale),
                },
            ),
            # Populated for the same reason `facts` is: `detail` is a sentence, and a surface
            # that wants to offer the action should not have to find it inside one. The
            # permissions check sets both and this is the other check with a command to give.
            remedy=f"manicule document list --source {first}",
        )

    async def _sessions_check(self) -> r.Check:
        """Whether the server holds a Confluence session for every source that needs one.

        **The check a restart makes necessary.** A session lives in the server's memory and
        nowhere else, so launchd restarting the process — a crash, a logout, a reboot — ends every
        one of them, and the next scheduled sync fails to authenticate at whatever hour the
        schedule comes round. Before this, the only trace of that was a line on the server's
        stderr; ``doctor`` reported a healthy installation, because from every angle it looked at,
        it was one.

        **It asks the server rather than answering for itself, and that is the whole difficulty.**
        ``manicule doctor`` runs in the process somebody typed it in — the operation is read-only,
        so it takes no lock and is not proxied — and that process holds no sessions and never
        will. Reading its own vault would report "no session" always: alarming, never
        informative, and quickly ignored. So the question goes over the control socket to the
        process that has the answer.

        Except when this *is* that process. Served manicule answers ``doctor`` over MCP and over
        the HTTP route, and there the vault in front of it is the one the syncs read, so it is
        read directly. The two are told apart by which vault holds something, which needs no flag
        and no second source of truth: nothing but a server ever holds a session, so a non-empty
        vault means this process is the server. A *server holding none* — the case this check is
        for — falls through to the socket and is answered, correctly, by itself.

        Four states, and none of them is ``failing``: nothing is broken, and a corpus that syncs
        by token or by filesystem is unaffected. A session that has to be re-taken is a thing to
        do rather than a fault to repair, which is the reading
        :meth:`_connectors_check` applies for the same reason.
        """
        from manicule.connectors.config import (  # noqa: PLC0415 - no HTTP stack at import
            CONNECTOR_NAME,
            AuthMethod,
            ConfluenceConfig,
        )
        from manicule.connectors.sessions import authority_key, default_store  # noqa: PLC0415

        wanted: dict[str, str] = {}
        for name, configured in sorted(self.settings.connectors.items()):
            if configured.type != CONNECTOR_NAME or not configured.enabled:
                continue
            try:
                config = ConfluenceConfig.model_validate(configured.options)
            except ValidationError:
                # Configuration that will not validate is `_configuration_check`'s finding and
                # not this one's. Reporting it twice, in different words, sends an operator to
                # fix a session for a source whose real problem is a setting.
                continue
            if config.auth_method is AuthMethod.BROWSER_SESSION:
                wanted[name] = authority_key(config.base_url)
        if not wanted:
            return r.Check(
                name="sessions",
                state="ok",
                detail="no configured source authenticates with a Confluence browser session",
                facts={"sources": []},
            )

        held = default_store().holding()
        where = "this process, which holds them"
        if not held:
            asked = await self._sessions_from_the_server()
            if asked is None:
                return r.Check(
                    name="sessions",
                    state="degraded",
                    detail=(
                        f"{_listed(wanted)} authenticate(s) with a Confluence browser session, "
                        f"and no manicule server is running to hold one. Nothing is synced on a "
                        f"schedule while there is no server. Start one with `manicule serve`, "
                        f"then sign in with "
                        f"`manicule connector login {sorted(wanted)[0]} --browser`."
                    ),
                    facts=cast(
                        "dict[str, JsonValue]",
                        {"sources": sorted(wanted), "held": [], "server": False},
                    ),
                    remedy="manicule serve",
                )
            held, where = asked, "the running server, over its control socket"

        missing = sorted(name for name, key in wanted.items() if key not in held)
        facts = cast(
            "dict[str, JsonValue]",
            {
                "sources": sorted(wanted),
                # The instances a session is held for, and never a cookie: this is the fact a
                # machine reading `--json` wants, and the account is in `detail` for the person
                # who wants to know *whose* session is signed in.
                "held": sorted(held),
                "missing": missing,
                "server": True,
                "read_from": where,
            },
        )
        if missing:
            first = missing[0]
            return r.Check(
                name="sessions",
                state="degraded",
                detail=(
                    f"the manicule server holds no Confluence session for {_listed(missing)}, so "
                    f"the next scheduled sync of {'it' if len(missing) == 1 else 'each'} stops "
                    f"without contacting the instance. This is what a restart looks like — "
                    f"sessions live in the "
                    f"server's memory and do not survive one — rather than a fault, and it is "
                    f"not an outage at the instance: nothing has been asked of it. Sign in again "
                    f"with `manicule connector login {first} --browser`; manicule never asks for "
                    f"the password and never sees it."
                ),
                facts=facts,
                remedy=f"manicule connector login {first} --browser",
            )
        signed_in = ", ".join(
            f"{name} as {held[key] or 'an unnamed account'}" for name, key in sorted(wanted.items())
        )
        return r.Check(
            name="sessions",
            state="ok",
            detail=f"a Confluence session is held for {signed_in}, read from {where}",
            facts=facts,
        )

    async def _sessions_from_the_server(self) -> dict[str, str] | None:
        """What the running server says it is holding, or ``None`` when there is no server.

        Every failure reaching a socket reads as "no server", and that is deliberate rather than
        swallowed: a socket that is there and unusable, a server one version behind that does not
        know the frame, a connection that dies — the operator's next step in all of them is the
        same as for a server that is not running, and ``manicule connector sync`` already reports
        the unusable-socket case in its own words with its own remedy. A diagnostic producing
        four sentences here would be four sentences for one thing to do.
        """
        from manicule.app import control  # noqa: PLC0415 - only this check crosses the socket

        try:
            path = control.socket_path(self.settings.data_dir)
            if not control.is_serving(path):
                return None
            answered = await control.connect(path, control.Held(), on_progress=_ignored)
        except (ManiculeError, ValueError, OSError):
            return None
        data = answered.get("data")
        if not answered.get("ok") or not isinstance(data, dict):
            return None
        rows = data.get("held")
        if not isinstance(rows, list):
            return None
        held: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            base_url = row.get("base_url")
            if isinstance(base_url, str):
                held[base_url] = _text(row.get("account"))
        return held

    async def _document_identity_check(self) -> r.Check:
        """Documents keyed on where they sit while the file beside them declares a page id.

        A local file's identity used to be its resolved path, always. It is now the ``source_id``
        a :mod:`~manicule.connectors.sidecar` manifest declares, where one does — so that a mirror
        reorganized from by-space to by-tree updates its pages instead of replacing every one of
        them with a new document. Documents ingested before that keep the old identity until the
        next sync, and the next sync moves them. This is the dry run that says so first.

        **It is a stronger position than #98's and the difference is worth stating.** That change
        had nothing in the database recording which connector instance a row came from, so a
        migration would have had to guess and none shipped. Here the mapping is a fact already
        stored: ``documents.source_id`` holds the old identity and the provenance record's own
        ``source_id`` holds the new one, written by the same fetch, for exactly the rows that
        move and no others. Every field below is read rather than estimated.

        **Chunks and vectors cannot be reused, and not only because the ids move.** ``chunk_id``
        derives from ``document_id``, so every chunk id changes — but even re-keyed they would be
        wrong, because the documents whose identity moves are precisely the documents whose
        *parse* changes: an enriched export stops going through the HTML parser and starts going
        through the storage parser, which produces different blocks and therefore different text.
        The re-embedding is unavoidable, so a migration would buy the curation rather than the
        compute, and this check says that plainly rather than implying a cheaper option exists.

        **``degraded``, not ``failing``**, on #98's reading: the documents are indexed, searchable
        and correctly cited, and every query over them works. What is true is that their identity
        is about to change.
        """
        try:
            store = await self._backend.documents()
            # `list_documents` rather than `select_documents`: this is the read every surface
            # uses, so the check runs against the same scoping and the same workspace filter a
            # listing does. The repair selector is a different surface with different predicates
            # and none of them is the one being asked about here.
            documents = await store.list_documents(limit=_IDENTITY_SCAN)
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="document-identity",
                state="unknown",
                detail=f"the corpus could not be examined: {type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        # What identities are actually in use, so that a page-keyed document is recognized as the
        # twin of the path-keyed one beside it rather than as an unrelated row.
        claimed = {document.source_id for document in documents}
        moving = [
            document
            for document in documents
            if (declared := _declared(document))
            and declared != document.source_id
            and not _identity_deliberately_unapplied(document, claimed=claimed)
        ]
        if not moving:
            return r.Check(
                name="document-identity",
                state="ok",
                detail=(
                    "no documents are indexed, so no identity can be about to change"
                    if not documents
                    # "None affected" and "none looked at past ten thousand" are different
                    # answers, and a check that gave the first when it meant the second would be
                    # a clean bill of health for a corpus nobody examined.
                    else "every document is already keyed on the identity its source declares"
                    if len(documents) < _IDENTITY_SCAN
                    else f"the first {_IDENTITY_SCAN} documents are already keyed on the "
                    f"identity their sources declare"
                ),
                # `truncated` on the ok path as well as the degraded one. The prose above already
                # distinguishes "none" from "none in the first ten thousand"; a caller reading
                # `facts` got `affected: 0` and no way to tell which sentence produced it.
                facts={
                    "affected": 0,
                    "examined": len(documents),
                    "truncated": len(documents) >= _IDENTITY_SCAN,
                },
            )
        first = moving[0]
        return r.Check(
            name="document-identity",
            state="degraded",
            detail=(
                f"{len(moving)} document(s) are still keyed on their file's path while the "
                f"manifest beside each one declares an identity of its own — {first.source_id!r} "
                f"declares {first.provenance.source.source_id!r}. "  # pyright: ignore[reportOptionalMemberAccess]
                f"Documents ingested before the identity change are re-keyed in place when this "
                f"database migrates, taking their chunks, versions, glossary entries, collection "
                f"membership and tags with them. These were left alone because their declared "
                f"identity is already held by another document, and moving onto an occupied key "
                f"would overwrite a row nothing can restore. Two rows now stand for one page and "
                f"the path-keyed one will never update, by whichever of two routes it got here: "
                f"where two files declare one id, each is re-indexed under its own path on every "
                f"sync; where a manifest applied the identity after the page had already been "
                f"indexed, the path-keyed row stops being discovered at all, so no sync refreshes "
                f"it and no sync removes it. Nothing is lost meanwhile — both are "
                f"indexed, searchable and correctly cited. Compare them with "
                f"`manicule document list --source {first.source}`, remove whichever is the "
                f"stale copy with `manicule document delete <id>`, or correct the manifest that "
                f"declares the wrong page id and sync again. There is no bulk remedy for this "
                f"and none is implied"
                # The list below is capped, and the cap is stated *here* rather than left to be
                # inferred from a length. `affected` and `len(documents)` disagreeing is how a
                # script that iterated the sample deletes some of the corpus and reports success
                # — the `documents` array is the only machine-readable enumeration this check
                # offers, so a caller has to be told when it is not the whole of it.
                + (
                    # `--limit` is named because it is not optional in practice: the listing
                    # defaults to 50, and a corpus with this many affected rows has more than
                    # that. Emitting the command without it was this message's own first draft,
                    # and running it returned 50 of 80 rows and 10 of 40 affected — the same
                    # class of untrue instruction the sidecar remediation was fixed for.
                    f". Listed below: {min(len(moving), _IDENTITY_SAMPLE)} of {len(moving)}. This "
                    f"diagnostic is a sample; `manicule document list --source {first.source} "
                    f"--limit {max(len(documents) * 2, 100)} --json` lists the corpus itself, and "
                    f"the affected rows are the ones whose provenance source_id differs from "
                    f"their source_id. Raise --limit past your corpus size — it defaults to 50"
                    if len(moving) > _IDENTITY_SAMPLE
                    else ""
                )
                + (
                    # Ordering, not remediation. At export scale this state is reached one row
                    # per page, and per-document removal is what there is; the cheap fix is to
                    # not arrive here, which is a thing to say before the next corpus rather than
                    # after this one.
                    ". Converting an export *before* its first sync avoids this entirely: the "
                    "manifest is read at the first discovery and no path-keyed row is ever "
                    "created"
                    if _sidecar_ordering_would_have_helped(moving)
                    else ""
                )
                + "."
            ),
            facts=cast(
                "dict[str, JsonValue]",
                {
                    "affected": len(moving),
                    "listed": min(len(moving), _IDENTITY_SAMPLE),
                    "examined": len(documents),
                    "truncated": len(documents) >= _IDENTITY_SCAN,
                    "chunks_reusable": False,
                    "vectors_reusable": False,
                    "citations_change": True,
                    "documents": [
                        {
                            "source": document.source,
                            "old_source_id": document.source_id,
                            "old_document_id": document.id,
                            "new_source_id": document.provenance.source.source_id,  # pyright: ignore[reportOptionalMemberAccess]
                            "new_document_id": document_id(
                                self.settings.workspace,
                                document.source,
                                document.provenance.source.source_id,  # pyright: ignore[reportOptionalMemberAccess]
                            ),
                            "uri": document.uri,
                        }
                        # Bounded, and the bound is named in `affected` rather than hidden: a
                        # corpus of ten thousand mirrored pages would otherwise put ten thousand
                        # rows into a diagnostic somebody runs to read a sentence.
                        for document in moving[:_IDENTITY_SAMPLE]
                    ],
                },
            ),
            remedy=f"manicule document list --source {first.source}",
        )

    async def _document_content_check(self) -> r.Check:
        """Documents whose identity was repaired and whose stored text has not been rebuilt yet.

        **The window this exists to make visible.** Re-keying a document moves its row; it does
        not re-read the file. So between the migration and the next sync a re-keyed enriched page
        has its correct identity and its *old chunks* — the generic-HTML parse of the wrapper,
        metadata banner and all. The corpus still returns exactly the content this whole change
        exists to keep out of it.

        Deleting the chunks in the migration would be worse: it removes content before its
        replacement exists, and a page that answers nothing is a worse answer than a page that
        answers stalely. So the intermediate state is kept and **stated**, which is the difference
        between a design and a trap. An operator who runs a migration, sees it succeed and stops
        would otherwise have a corpus with the original defect and nothing anywhere saying so.

        **It clears itself.** The migration records the ``content_hash`` it saw, and only for
        documents whose parse actually changes; this compares that against the live column, so the
        moment a sync re-ingests the page with its extracted body the two differ and the finding
        is gone. A document whose parse is unaffected — a mirrored PDF with a manifest — carries
        no marker and is never reported, because it owes nothing.
        """
        try:
            store = await self._backend.documents()
            documents = await store.list_documents(limit=_IDENTITY_SCAN)
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="document-content",
                state="unknown",
                detail=f"the corpus could not be examined: {type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        # The same bound the identity check reads, reported the same way. This check is the one
        # that had it nowhere: neither its prose nor its facts said the scan stopped, so a corpus
        # of a million rows reported "no document is serving stale text" having looked at one per
        # cent of it. Both states report it, because a bounded count is bounded whether it came
        # back zero or not.
        truncated = len(documents) >= _IDENTITY_SCAN
        bound = f" among the first {_IDENTITY_SCAN} documents examined" if truncated else ""
        waiting = [document for document in documents if _awaiting_reparse(document)]
        if not waiting:
            return r.Check(
                name="document-content",
                state="ok",
                detail=(
                    "no documents are indexed, so none can be serving text from before their "
                    "identity was repaired"
                    if not documents
                    else f"no document is serving text from before its identity was repaired{bound}"
                ),
                facts={
                    "awaiting_reparse": 0,
                    "examined": len(documents),
                    "truncated": truncated,
                },
            )
        sources = sorted({document.source for document in waiting})
        return r.Check(
            name="document-content",
            state="degraded",
            detail=(
                f"{len(waiting)} document(s){bound} were re-keyed onto the identity their source "
                f"declares and have not been re-read since. Their identity is correct and their "
                f"stored text is not: it is the parse from before the storage body was routed to "
                f"the storage parser, so it still carries the exporter's metadata banner and its "
                f"macros still read as prose. Nothing is broken and nothing is missing — the "
                f"pages are indexed, searchable and correctly cited — but they answer with text "
                f"this build would not produce. Run "
                f"{', '.join(f'`manicule connector sync {name}`' for name in sources)} to rebuild "
                f"it, or `manicule index <path>` for a directory that is not a configured source. "
                f"This clears itself once they are re-read."
            ),
            facts=cast(
                "dict[str, JsonValue]",
                {
                    "awaiting_reparse": len(waiting),
                    "examined": len(documents),
                    "truncated": truncated,
                    "sources": sources,
                    "documents": [
                        {"source": document.source, "source_id": document.source_id}
                        for document in waiting[:_IDENTITY_SAMPLE]
                    ],
                },
            ),
            remedy=f"manicule connector sync {sources[0]}",
        )

    async def _wiki_provenance_check(self) -> r.Check:
        """Live-wiki documents indexed before the connector recorded what the source said.

        **What is missing is one field, and it is the one that cannot be recovered locally.** A
        document ingested before this connector wrote provenance still holds its page id, its
        canonical URI, its fetched version, its media type and its hierarchy — six of the seven
        fields, all of them already in the row. It does not hold ``modified_at``, which was never
        stored anywhere, and which is precisely the field a durable citation needs and a local
        clock must never supply. So the record is not reconstructed here: a partial one, written
        with the timestamp left null, is indistinguishable at every surface from a page whose
        source genuinely reported no modification time. That is a worse outcome than the gap,
        because it is a gap that claims to be an answer.

        **A refetch is therefore required, and a routine sync will not perform one.** An
        unchanged page is unreachable twice over: the watermark means discovery never enumerates
        it, and the version-token comparison means it would skip without a fetch even if it were
        enumerated. ``manicule reindex`` cannot help either — it re-parses retained bytes and
        touches no network. So the honest instruction is a full resync of the source, and this
        check says that rather than implying something cheaper exists.

        **It reports and changes nothing.** ``doctor --fix`` seeds grammars and vocabularies:
        local, cheap, idempotent. Re-fetching a remote wiki is none of those, and arming it from
        a health command is exactly the recrawl-behind-a-routine-operation this repository's
        connector documentation warns against. The count is what an operator needs to decide;
        the decision stays theirs.

        ``degraded`` rather than ``failing``: every one of these documents is indexed,
        searchable and correctly cited from its ordinary fields. What they cannot do is carry a
        claim-level source citation.
        """
        # Local, like every other reference to this module in this file: it is the registered
        # name of the live connector, and importing it eagerly would pull connector
        # configuration into a process that may never sync anything.
        from manicule.connectors.config import CONNECTOR_NAME  # noqa: PLC0415

        live = sorted(
            name
            for name, connector in self.settings.connectors.items()
            if connector.type == CONNECTOR_NAME
        )
        if not live:
            # Plainly, and with no mention of records or migrations. Somebody running `doctor`
            # on an installation that has never pointed at a wiki is not an audience for either.
            return r.Check(
                name="wiki-provenance",
                state="ok",
                detail="no live Confluence source is configured",
            )
        try:
            store = await self._backend.documents()
            documents = await store.list_documents(limit=_IDENTITY_SCAN)
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="wiki-provenance",
                state="unknown",
                detail=f"the corpus could not be examined: {type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        # The same bound the identity and content checks read, reported the same way. A count
        # that stopped early and did not say so is a clean bill of health over one per cent of a
        # corpus, and both states report it because bounded is bounded either way.
        truncated = len(documents) >= _IDENTITY_SCAN
        bound = f" among the first {_IDENTITY_SCAN} documents examined" if truncated else ""
        # Only documents carrying **no record at all**. One that carries a stated refusal is a
        # different finding with a different repair — the source declared something a citation
        # will not render — and re-fetching it would produce the same refusal again.
        sources = set(live)
        missing = [
            document
            for document in documents
            if document.source in sources and document.provenance is None
        ]
        if not missing:
            return r.Check(
                name="wiki-provenance",
                state="ok",
                detail=(
                    f"every live Confluence document carries an authoritative source record{bound}"
                ),
                facts=cast(
                    "dict[str, JsonValue]",
                    {"sources": live, "examined": len(documents), "truncated": truncated},
                ),
            )
        affected = sorted({document.source for document in missing})
        return r.Check(
            name="wiki-provenance",
            state="degraded",
            detail=(
                f"{len(missing)} document(s) from {', '.join(affected)} were indexed before this "
                f"connector recorded what the source said about them{bound}. They are searchable "
                f"and correctly cited from their own fields; what they cannot supply is a "
                f"claim-level citation carrying the page's version and modification time. The "
                f"modification time was never stored locally and must not be invented, so these "
                f"pages have to be fetched again. **A routine incremental sync will not do it**: "
                f"the stored watermark means discovery does not enumerate a page nobody has "
                f"edited, and its version token is unchanged, so it would skip without a fetch "
                f"even if it were enumerated. Re-sync {affected[0]} in full."
            ),
            facts=cast(
                "dict[str, JsonValue]",
                {
                    "missing_provenance": len(missing),
                    "sources": affected,
                    "examined": len(documents),
                    "truncated": truncated,
                    "documents": [
                        {"source": document.source, "source_id": document.source_id}
                        for document in missing[:_IDENTITY_SAMPLE]
                    ],
                },
            ),
            remedy=f"resync {affected[0]} in full — see docs/connectors/confluence.md §2.2",
        )

    async def _index_check(self) -> r.Check:
        try:
            store = await self._backend.documents()
            fingerprints = await store.index_fingerprints()
            documents = await store.count_documents()
            ingestion = await self._backend.ingestion()
            inspect_physical = getattr(ingestion, "physical_index_fingerprint", None)
            physical_embed = (
                await inspect_physical()
                if fingerprints.is_empty and inspect_physical is not None
                else None
            )
            if fingerprints.is_empty and physical_embed is None:
                configured_embed = configured_chunk = ""
            else:
                configured_embed, configured_chunk = await ingestion.configured_index_fingerprints()
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="index",
                state="failing",
                detail=f"{type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        if fingerprints.is_empty:
            if documents == 0 and physical_embed is not None and physical_embed != configured_embed:
                return r.Check(
                    name="index",
                    state="degraded",
                    detail=(
                        "the empty workspace retains physical vector fingerprint metadata "
                        "without a matching relational index identity"
                    ),
                    facts={
                        "documents": 0,
                        "vector_table": None,
                        "stale_empty_identity": True,
                        "physical_fingerprint_mismatch": True,
                    },
                    remedy="manicule reset-index --yes",
                )
            return r.Check(
                name="index",
                state="ok" if documents == 0 else "degraded",
                detail=(
                    "empty index, ready for a first ingest"
                    if documents == 0
                    else f"{documents} document(s) but no recorded fingerprints"
                ),
                # No remedy. "Documents but no fingerprints" is a state whose repair depends on
                # how it was reached, and a `remedy` naming a command with a `<path>` in it is
                # neither runnable by a script nor certain to be the right advice.
                facts={"documents": documents, "vector_table": None},
            )
        embed_mismatch = bool(
            fingerprints.embed is not None
            and configured_embed
            and fingerprints.embed.canonical() != configured_embed
        )
        chunk_mismatch = bool(
            fingerprints.chunk is not None
            and configured_chunk
            and fingerprints.chunk.canonical() != configured_chunk
        )
        if documents == 0 and (embed_mismatch or chunk_mismatch):
            return r.Check(
                name="index",
                state="degraded",
                detail=(
                    "the empty workspace retains an obsolete derived fingerprint that would "
                    "reject the configured first ingest"
                ),
                facts={
                    "documents": 0,
                    "vector_table": fingerprints.vector_table or None,
                    "stale_empty_identity": True,
                },
                remedy="manicule reset-index --yes",
            )
        return r.Check(
            name="index",
            state="ok",
            detail=f"{documents} document(s) in {fingerprints.vector_table or 'no vector table'}",
            facts={"documents": documents, "vector_table": fingerprints.vector_table or None},
        )

    async def _glossary_check(self) -> r.Check:
        """Whether the corpus's definitions came from the detector that is installed.

        **The check that makes a clean ``doctor`` mean what a reader assumes it means.** Before
        it, every fingerprint this command reported was about parsing, chunking or embedding,
        so an index whose glossary was three detector versions out of date passed with nothing
        amber anywhere — and the assumption a green result invites is precisely that the index
        is current, not that three of its four stages are.

        **``degraded``, not ``failing``, and the same reading the document-content check
        applies.** Nothing is broken: the documents are indexed, the definitions are cited to
        passages that resolve, and every query works. What is true is that some of those
        definitions were produced by rules this build has since corrected, which is a thing to
        act on rather than a fault to repair. The remedy is one command, it touches no network
        and no model, and it is named.

        **Detection switched off is ``ok``, and reported as the deliberate state it is.** An
        operator who has turned the detector off while investigating it does not need a health
        check telling them so in amber every time they run it; what they need is for the state
        to be visible, which is what the detail line does.
        """
        try:
            ingestion = await self._backend.ingestion()
            detector = await ingestion.glossary_fingerprint()
            store = await self._backend.documents()
            # A count over an indexed column. `doctor` is run to read a sentence, and a check
            # that read the corpus's vocabulary to decide whether the vocabulary is current
            # would cost the thing it is asking about.
            stale = await store.count_documents(glossary_fp_other_than=detector.canonical())
            unversioned = await store.count_documents(glossary_fp_unrecorded=True)
            documents = await store.count_documents()
        except Exception as exc:  # noqa: BLE001 - the exception is the diagnosis
            return r.Check(
                name="glossary",
                state="unknown",
                detail=f"the corpus could not be examined: {type(exc).__name__}: {exc}",
                facts={"error_type": type(exc).__name__},
            )
        facts: dict[str, JsonValue] = {
            "detector": detector.describe(),
            "enabled": detector.detects,
            "documents": documents,
            "stale": stale,
            "unversioned": unversioned,
        }
        if not detector.detects:
            return r.Check(
                name="glossary",
                state="ok",
                detail=(
                    "glossary detection is switched off "
                    "(rag.glossary.detect_on_ingest = false), so definitions already stored "
                    "stay queryable and no new ones are read"
                ),
                facts=facts,
            )
        if not stale:
            return r.Check(
                name="glossary",
                state="ok",
                detail=f"every document's glossary came from {detector.describe()}",
                facts=facts,
            )
        # Two sentences, because they send a reader to two different conclusions. An index that
        # predates the column has never had glossary lineage at all and needs a one-time
        # migration — a fact about the upgrade rather than about the detector. An index that has
        # it and disagrees is ordinary staleness after a rule change. Reporting the first as the
        # second tells somebody their detector moved when what actually happened is that they
        # upgraded, and the number would look alarming in proportion to how long they have been
        # running manicule.
        if unversioned == stale:
            detail = (
                f"{unversioned} of {documents} document(s) carry no glossary lineage at all, "
                f"which is what an index created before glossary state was versioned looks "
                f"like. Their definitions are whatever detector produced them and nothing "
                f"recorded which. This is a one-time migration rather than a fault: run "
                f"`manicule document reindex --stale-glossary`, which recomputes them from the "
                f"chunks already stored and runs no parser, no connector and no embedder."
            )
        else:
            detail = (
                f"{stale} of {documents} document(s) hold glossary entries the installed "
                f"detector did not produce"
                + (f", {unversioned} of them never versioned at all" if unversioned else "")
                + ". Their definitions are cited and resolvable and may simply be right — but "
                "they came out of rules this build has changed, and nothing else notices: a "
                "detector change moves no parse, chunk or embedding fingerprint, so a sync "
                "skips these documents and every other check here passes. Run "
                "`manicule document reindex --stale-glossary`, which reads stored chunks and "
                "runs no parser, no connector and no embedder."
            )
        return r.Check(
            name="glossary",
            state="degraded",
            detail=detail,
            facts=facts,
            remedy="manicule document reindex --stale-glossary",
        )

    async def _grammar_check(self, *, fix: bool = False) -> r.Check:
        """Whether the declared code grammars are on this machine — and, with ``fix``, seed them.

        The grammars are **not in the grammar pack's wheel**: it downloads them per language on
        first use, so a fresh install has none and every source file it is offered is refused
        (``docs/parsing.md`` §8.1). That refusal is deliberate — a silent line-splitting
        fallback is how two machines end up with two chunkings of one corpus — but it is only
        actionable if something says so before a person indexes their code and finds every
        ``.py`` file marked unsupported. This is that something.

        **Degraded rather than failing when grammars are absent**, and the distinction is a
        judgment about what a bad answer means. Nothing here is broken: an installation whose
        corpus is Markdown and PDFs works perfectly with no grammars at all, and reporting a
        red check on it would teach an operator to ignore ``doctor``. What is true is that
        source files will be refused, which is a capability this installation does not
        currently have. The two states that *are* failing are a bundle that is present and
        unusable, and a declared set that does not validate: both are misconfigurations
        somebody made, and both are invisible until something tries to parse code.

        **The configured cache is applied before the question is asked**, which is the whole
        reason this is not two lines. The grammar pack keeps one process-global registry and
        ``doctor`` runs before any parser is constructed, so a check that simply asked would
        answer about the per-user cache while the installation reads an image-local one — and
        would report an image full of grammars as empty. That is the same shape as the macOS
        cache-redirect bug this area has already been burned by, so the configuration is
        applied here explicitly, with the values the parser itself would use. It is the same
        call the parser's own factory makes with the same values read from the same settings,
        so a ``doctor`` run alongside an ingest repoints nothing: the cost of the overlap is
        that the pack's per-language parser cache is rebuilt, not that it is rebuilt differently.

        Args:
            fix: Seed what is missing before reporting, from the offline bundle first and the
                network only for what the bundle did not supply. A failed seed is ``failing``
                rather than ``degraded``: a repair was asked for and did not happen.

        Returns:
            One check. It never raises — every way this can go wrong is a diagnosis, and a
            ``doctor`` that crashed on a broken bundle would be a diagnostic that only works
            on a healthy machine.
        """
        return await asyncio.to_thread(self._inspect_grammars, fix=fix)

    def _inspect_grammars(self, *, fix: bool) -> r.Check:
        """The blocking half of :meth:`_grammar_check`, off the event loop.

        Blocking because it lists a directory, reads a bundle manifest, and — under ``fix`` —
        copies libraries or fetches a release. The import is lazy for the reason every parsing
        import in this file is: an installation without the parsing extra still runs ``doctor``,
        and it must not be the command that discovers the extra is missing by failing to start.
        """
        from manicule.parsers import grammar_bundle, grammars  # noqa: PLC0415 - a parsing extra

        try:
            config = self._source_code_config()
        except ValidationError as exc:
            return r.Check(
                name="grammars",
                state="failing",
                detail=(
                    f'the code parser\'s configuration (plugins.config."parser.sourcecode") '
                    f"does not validate, so nothing can say which grammars this installation "
                    f"needs: {exc}"
                ),
            )

        seeded: tuple[str, ...] = ()
        try:
            languages = grammars.validate_languages(config.languages)
            grammars.configure_pack(
                languages,
                cache_dir=config.grammar_cache_dir,
                manifest_url=config.grammar_manifest_url,
            )
            # `locate` rather than `resolve`: it answers "is there a bundle here at all" from
            # an environment variable and a module spec, where resolving reads the manifest and
            # stats every library in it. That matters because the browser surface renders this
            # check on a page, and `bundle_status` below reads the bundle anyway — so resolving
            # here would read it twice per request to say one thing.
            located = grammar_bundle.locate()
            missing = grammars.missing_grammars(languages)
            if missing and fix:
                seeded = grammars.prefetch(languages)
                # Asked of the cache again rather than taken from what the seed returned. The
                # pack's own prefetch reports success for a language in its process-global
                # registry whatever the configured cache holds, which is how an image ships
                # with no grammars in it and is told it succeeded.
                missing = grammars.missing_grammars(languages)
            # Quoted when it is actionable — something is absent, or a bundle is installed and
            # is therefore a fact about this machine worth stating. On a healthy install with no
            # bundle it is three lines of advice about a thing nobody needs, and this check is
            # read on a page. A bundle that is present and unusable raises out of here, which is
            # the one case where "no bundle" would be a lie.
            offline = grammars.bundle_status() if missing or located is not None else ""
        except ImportError:
            return r.Check(
                name="grammars",
                state="unknown",
                detail=(
                    "the parsing extra is not installed, so there are no grammars to check "
                    "and no source file would be parsed here. Install `manicule[parsers]`."
                ),
            )
        except ManiculeError as exc:
            # ConfigError for a language key that is not in the manifest, GrammarBundleError
            # for a bundle that is installed and wrong, GrammarFetchError for a seed that
            # could not complete. Each already says what to do about itself.
            return r.Check(name="grammars", state="failing", detail=str(exc))

        cache = grammars.cache_directory()
        if missing:
            # The action first, then the detail, and the offline-bundle paragraph only when a
            # bundle is actually installed. On a fresh machine this fired with the fix buried
            # mid-sentence between all twenty-four language names and three lines about
            # building a bundle on another host — advice for an air-gapped operator, shown to
            # everyone who had simply not run `init` yet. A host with no route to anything
            # reaches the *failing* branch above, whose message carries the search path.
            over = len(missing) - _LANGUAGES_NAMED
            named = ", ".join(missing[:_LANGUAGES_NAMED]) + (
                f" and {over} more" if over > 0 else ""
            )
            carried = f" — {offline}" if located is not None else ""
            return r.Check(
                name="grammars",
                state="degraded",
                detail=(
                    f"run `manicule doctor --fix` to seed {len(missing)} missing grammar(s) "
                    f"into {r.redacted_path(cache)}: {named}. Until then a document in one of "
                    f"those languages is refused rather than line-split{carried}"
                ),
                facts={
                    "cache": r.redacted_path(cache),
                    "missing": list(missing),
                    "declared": len(languages),
                },
                remedy="manicule doctor --fix",
            )
        settled = f"seeded {list(seeded)}; " if seeded else ""
        carried = f"; {offline}" if offline else ""
        return r.Check(
            name="grammars",
            state="ok",
            detail=(
                f"{settled}{len(languages)} declared grammar(s) in {r.redacted_path(cache)} "
                f"({grammars.PACK_DISTRIBUTION} {grammars.pack_version()}){carried}"
            ),
            facts={
                "cache": r.redacted_path(cache),
                "missing": [],
                "declared": len(languages),
                "seeded": list(seeded),
            },
        )

    async def _vocabulary_check(self, *, fix: bool = False) -> r.Check:
        """Whether the BPE vocabularies this installation counts tokens with are on it.

        ``tiktoken`` ships none in its wheel: every encoding it knows is a file on a blob
        store, fetched on first use (``docs/retrieval.md`` §7.2). manicule will not make that
        fetch while answering a question, so an installation that never pre-seeded is one that
        indexes a corpus perfectly and then refuses every ``search`` — the failure split across
        two moments, the second one unexplained. This is what says so at the first moment.

        **Failing rather than degraded, and the difference from :meth:`_grammar_check` is the
        whole reason to state it.** Missing grammars are degraded because there are
        installations for which their absence costs nothing: a corpus of Markdown and PDFs
        works perfectly without one, and a red check on a healthy machine teaches an operator
        to ignore ``doctor``. No such installation exists here. Every context is measured with
        this vocabulary whatever the corpus is made of, so a machine without it cannot answer
        a question at all — this is not a healthy machine being marked red, it is a broken one
        being marked broken, and the objection the grammar check is answering does not arise.

        The other two failing states are the same two, for the same reasons: a bundle that is
        present and unusable, and a configured encoding no installed ``tiktoken`` defines —
        both misconfigurations somebody made, and both invisible until a query.

        Args:
            fix: Seed what is missing before reporting, from the offline bundle first and the
                network only for what the bundle did not supply.

        Returns:
            One check. It never raises: a ``doctor`` that crashed on a broken bundle would be
            a diagnostic that only works on a healthy machine.
        """
        return await asyncio.to_thread(self._inspect_vocabularies, fix=fix)

    def _inspect_vocabularies(self, *, fix: bool) -> r.Check:
        """The blocking half of :meth:`_vocabulary_check`, off the event loop.

        Blocking, and not trivially: answering "is every vocabulary here" reads each file and
        checks it against the digest ``tiktoken`` declares, because a cache entry with the
        right name and the wrong bytes is one ``tiktoken`` deletes and re-fetches — which on a
        host that cannot fetch is a refusal that an existence check would have called healthy.
        That is 5 MB of hashing; under ``fix`` it may also copy from a bundle or download.

        The import is lazy for the reason every optional import in this file is: an
        installation without the retrieval extra still runs ``doctor``, and this must not be
        the check that discovers the extra is missing by failing to start.
        """
        from manicule import vocabularies  # noqa: PLC0415 - a retrieval extra
        from manicule.vocabularies import bundle as vocabulary_bundle  # noqa: PLC0415

        seeded: tuple[str, ...] = ()
        try:
            wanted = vocabularies.required_encodings(self.settings.rag.context.encoding)
            # `locate` rather than `resolve`, for the reason `_inspect_grammars` gives: it
            # answers "is there a bundle here at all" from an environment variable and a module
            # spec, where resolving reads the manifest — and `bundle_status` below reads it
            # anyway when there is something worth saying.
            located = vocabulary_bundle.locate()
            missing = vocabularies.missing_vocabularies(wanted)
            if missing and fix:
                seeded = vocabularies.prefetch(wanted)
                # Asked of the cache again rather than taken from what the pre-seed returned.
                # `prefetch` checks itself, and a check that trusted the repair it had just
                # asked for would report success for the one failure this area exists to catch.
                missing = vocabularies.missing_vocabularies(wanted)
            offline = vocabularies.bundle_status() if missing or located is not None else ""
        except ImportError:
            return r.Check(
                name="vocabularies",
                state="unknown",
                detail=(
                    "the retrieval extra is not installed, so there is no token counter here "
                    "to give a vocabulary to. Install `manicule[retrieval]`."
                ),
            )
        except ManiculeError as exc:
            # ConfigError for a configured encoding no installed tiktoken defines — a model
            # name in `rag.context.encoding` is the specific mistake docs/retrieval.md §7.2
            # asks the fitter never to accept — VocabularyBundleError for a bundle that is
            # installed and wrong, VocabularyFetchError for a pre-seed that could not
            # complete. Each already says what to do about itself.
            return r.Check(name="vocabularies", state="failing", detail=str(exc))

        cache = vocabularies.cache_directory()
        where = r.redacted_path(cache)
        if missing:
            return r.Check(
                name="vocabularies",
                state="failing",
                detail=(
                    f"no vocabulary for {list(missing)} in {where}, so every search refuses "
                    f"rather than downloading one mid-question. Run `manicule doctor --fix` "
                    f"to seed them — {offline}"
                ),
                facts={"cache": where, "missing": list(missing), "wanted": list(wanted)},
                remedy="manicule doctor --fix",
            )
        settled = f"seeded {list(seeded)}; " if seeded else ""
        carried = f"; {offline}" if offline else ""
        here = (
            f"{settled}{len(wanted)} vocabulary(ies) {list(wanted)} in {where} "
            f"(tiktoken {vocabularies.tiktoken_version()})"
        )
        # Present, and in a directory the operating system reclaims. `ok` there would be a
        # check reporting health about a machine that will refuse every question the week
        # after a temp sweep, with nothing having changed and nothing having said so — a check
        # whose name outruns what it verifies. Degraded rather than failing: it works today,
        # and the remedy is a setting rather than a repair.
        if vocabularies.is_impermanent(cache):
            return r.Check(
                name="vocabularies",
                state="degraded",
                detail=(
                    f"{here}, which is under the system temporary directory and is reclaimed "
                    f"on a schedule. Everything works until it is swept, and then every "
                    f"search refuses. Point {vocabularies.CACHE_DIR_ENV} at a durable "
                    f"directory — with it unset manicule uses "
                    f"{r.redacted_path(vocabularies.default_cache_directory())}{carried}"
                ),
                facts={"cache": where, "missing": [], "wanted": list(wanted), "durable": False},
                remedy=(
                    f"{vocabularies.CACHE_DIR_ENV}="
                    f"{r.redacted_path(vocabularies.default_cache_directory())}"
                ),
            )
        return r.Check(
            name="vocabularies",
            state="ok",
            detail=f"{here}{carried}",
            facts={
                "cache": where,
                "missing": [],
                "wanted": list(wanted),
                "durable": True,
                "seeded": list(seeded),
            },
        )

    async def _model_check(self, *, provider: str | None = None) -> r.Check:
        """Whether the embedding weights are on this machine, and what it costs if not.

        **The one artifact nothing pre-seeds, and this is the check that says so.** Grammars
        and vocabularies are seeded by ``init`` because they are megabytes; the weights are
        1.1 GB on MLX and 2.3 GB as an ONNX export, and refusing to work until somebody has
        downloaded one would turn "install manicule and point it at a directory" into an
        errand. So the download happens on the path that needs it — inside the first ``index``
        — and the whole cost of that decision lands in one place: a first ingest that pauses
        for minutes with a third-party progress bar and no explanation. This is the sentence
        that makes the pause expected instead of a hang.

        **``ok`` when they are absent, which is the point rather than a lapse.** A machine
        that has not downloaded a model yet is not a broken machine, and a red check on a
        healthy install is how an operator learns to stop reading ``doctor``. There is exactly
        one state here that is genuinely broken and it is the one that is failing: weights
        absent on an install that has told the hub it may not look, which cannot answer a
        question and will not say so until somebody asks one.

        **It never touches the network** — see :func:`manicule.embedding.runtimes.hub.is_cached`.
        A diagnostic that fetched a gigabyte to report on a gigabyte would be absurd.

        Args:
            provider: The embedder to report on, for the caller that has just chosen one and
                not yet reloaded configuration — ``init``. Defaults to the configured one.

        Returns:
            One check. It never raises: like every other check here, a broken installation has
            to be diagnosable by the command that diagnoses it.
        """
        return await asyncio.to_thread(self._inspect_models, provider=provider)

    def _embedder_config_model(self, provider: str) -> type[BaseModel] | None:
        """The configuration model the named embedder registered, or ``None``.

        ``None`` means "nothing installed provides this embedder, or it declared no
        configuration model" — both of which leave the caller with no settings to validate and
        nothing to say about the artifact beyond the defaults.
        """
        discovery = self._backend.discovery
        if discovery is None:
            return None
        try:
            record = discovery.registry.record(keys.EMBEDDER.named(provider))
        except UnknownComponentError:
            return None
        return record.config_model

    def _weights_plan(self, provider: str | None) -> WeightsPlan | None:
        """The artifact the configured backend will load, or ``None`` with no embedding extra."""
        from manicule.embedding.artifacts import (  # noqa: PLC0415 - an extra
            WeightsPlan,
            builtin_model_revision,
            planned_weights,
        )
        from manicule.embedding.cards import CARD_FILES  # noqa: PLC0415 - an extra
        from manicule.embedding.config import EmbedderConfig  # noqa: PLC0415 - an extra
        from manicule.embedding.runtimes.hub import (  # noqa: PLC0415 - an extra
            cached_revision,
        )

        settings = self.settings
        chosen = provider or settings.embedding.provider
        model = settings.embedding.model
        configured_model_revision = settings.embedding.revision
        model_revision = configured_model_revision or builtin_model_revision(model)
        if Path(model).expanduser().is_dir() and model_revision is not None:
            error = (
                f"embedding.model {model!r} is local, so embedding.revision cannot identify "
                "it. Remove the revision; local inputs use a content digest."
            )
            return WeightsPlan(
                provider=chosen,
                repo=model,
                patterns=(),
                present=False,
                error=error,
            )
        weights = ""
        try:
            raw = settings.component_config("embedder", chosen)
            # Validate against the model the backend **registered**, not against a name this
            # function knows. Backends ship their own `EmbedderConfig` subclass from their own
            # distribution — `manicule-mlx` adds a Metal cache bound — and `extra="forbid"`
            # means validating one of those against the base model fails on its own setting. An
            # earlier version branched on the literals "mlx" and "onnx", so it could only ever
            # be right about the backends that happened to be in-tree the day it was written.
            #
            # When the registry cannot answer — no discovery, or an embedder nothing installed
            # provides — fall back to the two fields this function actually reads rather than
            # skipping validation. Those live on the base model, so a mutable `weights_revision`
            # is still refused; what is given up is only the rejection of a *foreign* key, which
            # is a different diagnostic's job. Skipping instead would make `doctor` quietly
            # weaker exactly where it has least information.
            model_for_config = self._embedder_config_model(chosen)
            if model_for_config is None:
                base = set(EmbedderConfig.model_fields)
                config: EmbedderConfig = EmbedderConfig.model_validate(
                    {key: value for key, value in raw.items() if key in base}
                )
            else:
                validated = model_for_config.model_validate(raw)
                if not isinstance(validated, EmbedderConfig):
                    # A registered model that is not an `EmbedderConfig` cannot carry
                    # `weights`/`weights_revision`, so there is no plan to make from it.
                    return planned_weights(chosen, model, model_revision=model_revision)
                config = validated
            weights = config.weights
            revision = config.weights_revision
            if (
                not Path(model).expanduser().is_dir()
                and builtin_model_revision(model) is None
                and not (
                    model_revision
                    and len(model_revision) == _IMMUTABLE_COMMIT_LENGTH
                    and all(character in "0123456789abcdef" for character in model_revision)
                )
            ):
                model_revision = cached_revision(model, CARD_FILES, model_revision)
            return planned_weights(
                chosen,
                model,
                model_revision=model_revision,
                override=weights,
                revision=revision,
            )
        except ImportError:
            return None
        except (ConfigError, ValidationError) as exc:
            return WeightsPlan(
                provider=chosen,
                repo=weights or settings.embedding.model,
                patterns=(),
                present=False,
                error=str(exc),
            )

    def _inspect_models(self, *, provider: str | None) -> r.Check:
        """The blocking half of :meth:`_model_check`: one cache probe, off the event loop."""
        return _weights_check(self._weights_plan(provider))

    def _source_code_config(self) -> SourceCodeConfig:
        """The code parser's configuration, validated the way the parser's own factory does.

        Read from settings rather than defaulted, because the declared language set and both
        grammar overrides are configuration: a container image points the cache inside the
        image and an air-gapped site points the manifest at a mirror, and a diagnostic that
        checked manicule's defaults would be describing a different installation.
        """
        from manicule.parsers.config import SourceCodeConfig as Model  # noqa: PLC0415 - see above

        return Model.model_validate(self.settings.plugins.config.get("parser.sourcecode", {}))

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
        for part in _config_key_parts(key):
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
        parts = _config_key_parts(key)
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
        def mutate(document: dict[str, Any]) -> None:
            _assign(document, parts, parsed)
            _validate_structural_chunker_config(document, parts)

        await asyncio.to_thread(_update_config, path, mutate)
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

    async def export_corpus(
        self, target: Path | str, *, allow_insecure_target: bool = False
    ) -> r.ExportReport:
        """Write a portable archive: retained bytes and metadata, never chunks or vectors.

        An archive that carried chunks would carry an index built by another chunker and
        another embedder, and dropping it into a store whose fingerprints say otherwise is
        the silent-mismatch failure the fingerprints exist to prevent. The importing
        installation re-derives both.

        A group- or world-readable target is refused, with the same escape hatch ``backup``
        has and for the same reason it has one: an operator writing an archive onto a volume
        whose permissions are somebody else's decision has a real errand, and refusing it
        outright only moves the copy to a `cp -r` that nothing checks and nothing records.
        """
        maintenance = await self._backend.maintenance()
        documents, chunks = await maintenance.export_corpus(
            _local(target), allow_insecure_target=allow_insecure_target
        )
        return r.ExportReport(path=str(_local(target)), documents=documents, chunks=chunks)

    async def import_corpus(self, source: Path | str, *, force: bool = False) -> r.IngestReport:
        """Ingest an exported archive, re-deriving chunks and vectors here."""
        started = time.monotonic()
        path = _local(source)
        if not await asyncio.to_thread(path.exists):
            msg = f"no such archive: {path}"
            raise UnknownEntityError(msg)
        ingestion = await self._backend.ingestion()
        payload = _ingest_payload(await ingestion.import_archive(path, force=force), started)
        if (
            payload.retry_required
            and payload.incomplete_reason is not None
            and payload.incomplete_reason.type == "CapacityRefusedError"
            and not force
        ):
            payload = payload.model_copy(
                update={
                    "incomplete_reason": payload.incomplete_reason.model_copy(
                        update={
                            "hint": (
                                "Free durable ingest capacity or raise the configured limit, "
                                "then retry this archive import with force enabled."
                            )
                        }
                    )
                }
            )
        return payload

    async def reset_index(self) -> r.ResetReport:
        """Delete derived chunks and vectors while retaining durable source snapshots.

        Disruptive until an offline rebuild republishes search state, so the CLI requires
        ``--yes``. The MCP and HTTP forms expose only the aggregate dry run.
        """
        maintenance = await self._backend.maintenance()
        outcome = await maintenance.reset_index()
        return r.ResetReport(
            documents=outcome.documents,
            chunks=outcome.chunks,
            vectors_removed=bool(
                outcome.vector_rows or outcome.publications or outcome.vector_store_removed
            ),
            vector_rows_removed=outcome.vector_rows,
            publications_removed=outcome.publications,
            memberships_removed=outcome.memberships,
            generations_terminalized=outcome.generations_terminalized,
            vector_store_removed=outcome.vector_store_removed,
            fingerprints_cleared=outcome.fingerprints_cleared,
            runtime_cache_invalidated=outcome.runtime_cache_invalidated,
            snapshots_retained=outcome.snapshots_retained,
        )

    async def lifecycle_reset_derived(self, *, dry_run: bool = False) -> r.LifecycleReport:
        maintenance = await self._backend.maintenance()
        if dry_run:
            return _lifecycle_plan_report(await maintenance.plan_reset_derived())
        return _lifecycle_outcome_report(await maintenance.reset_derived())

    async def lifecycle_cleanup_generations(self, *, dry_run: bool = False) -> r.LifecycleReport:
        maintenance = await self._backend.maintenance()
        if dry_run:
            return _lifecycle_plan_report(await maintenance.plan_derived_generation_cleanup())
        return _lifecycle_outcome_report(await maintenance.cleanup_derived_generations())

    async def lifecycle_release_history(
        self, cutoff: datetime, *, dry_run: bool = False
    ) -> r.LifecycleReport:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise PolicyError("source history retention cutoff must include a UTC offset")
        maintenance = await self._backend.maintenance()
        if dry_run:
            return _lifecycle_plan_report(await maintenance.plan_source_history_release(cutoff))
        return _lifecycle_outcome_report(await maintenance.release_source_history(cutoff))

    async def lifecycle_delete_snapshot(
        self,
        run_id: str,
        *,
        confirmation: str | None = None,
        dry_run: bool = True,
    ) -> r.LifecycleReport:
        maintenance = await self._backend.maintenance()
        if dry_run:
            return _lifecycle_plan_report(await maintenance.plan_snapshot_deletion(run_id))
        if confirmation is None:
            raise PolicyError(
                "snapshot deletion requires the confirmation token printed by its dry run"
            )
        return _lifecycle_outcome_report(
            await maintenance.delete_snapshot(run_id, confirmation=confirmation)
        )

    async def initialize(self, *, force: bool = False) -> r.InitReport:
        """Write a starting configuration, choosing what this machine can actually run.

        The hardware probe picks the embedding backend rather than recommending one, because
        a default that is wrong for the machine is discovered at the first ingest, and by then
        a corpus has been indexed with it.

        **It also pre-seeds the two artifacts no wheel ships**, through the same calls ``doctor
        --fix`` makes, so ``init`` and the repair cannot come to mean different things. The
        grammar pack fetches grammars per language on first use, which would make a file chunk
        one way on a machine that reached the network and another way on a machine that did
        not; ``tiktoken`` fetches its BPE vocabularies the same way, and without one no
        ``search`` can measure a context at all. ``docs/parsing.md`` §8.1 and
        ``docs/retrieval.md`` §7.2 close both by pre-seeding at install time, and ``init`` is
        install time. Each consults its offline bundle first, so a host carrying them needs no
        network here at all.

        A pre-seed that fails is **reported, not raised.** The configuration file is written by
        then, so raising would leave a config on disk and a command that says it failed, and the
        retry would need ``--force``; the notes carry the whole failure, ``doctor`` reports it
        afterwards, and the refusal each absence produces names the command that fixes it. An
        installation with no grammars is a real state — a Markdown corpus never needs one — and
        an installation that cannot reach a blob store is one somebody has to be *told* about
        rather than one this prevents by refusing to finish writing their configuration.
        """
        path = config_file()
        if await asyncio.to_thread(path.exists) and not force:
            msg = f"{path} already exists. Pass force to overwrite it."
            raise ConfigError(msg)
        probe = hardware()
        settings = self.settings
        provider = "mlx" if probe["apple_silicon"] else "onnx"
        await asyncio.to_thread(_update_config, path, _initial_configuration(settings, provider))
        pre_seed = await self._grammar_check(fix=True)
        vocabulary_seed = await self._vocabulary_check(fix=True)
        # Reported, never fetched — and that is a decision rather than an omission. The
        # weights are the one artifact `init` does not seed: they are a gigabyte and upward,
        # and an install step that downloads one before anybody has decided to index anything
        # makes a worse first five minutes than the download itself does. What was missing was
        # the sentence saying it is still to come, and this is that sentence.
        weights = await asyncio.to_thread(self._weights_plan, provider)
        weights_check = _weights_check(weights)
        notes = [
            f"embedding backend {provider!r} chosen for {probe['machine']} on {probe['system']}",
            "bind host written as 127.0.0.1; a wider bind needs auth and an explicit flag",
            f"grammars ({pre_seed.state}): {pre_seed.detail}",
            f"vocabularies ({vocabulary_seed.state}): {vocabulary_seed.detail}",
            f"weights ({weights_check.state}): {weights_check.detail}",
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
            # A field rather than something a renderer sniffs out of the note above it. The
            # note is one dim line among five and the pending download is the single fact that
            # changes what the next command feels like, so the surfaces are told it plainly
            # and each decides how loudly to say it. Read off the same plan the note above was
            # built from — one cache probe, two readers — rather than parsed back out of the
            # note: a boolean recovered from prose is a boolean that changes when somebody
            # rewords a sentence.
            weights_pending=weights is not None and not weights.present,
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
            # Named here and created by whoever writes into it. The service touches no
            # filesystem of its own: `backup` makes the directory and the one above it,
            # both 0700, and a call that never reaches storage — a fake backend, a dry run —
            # must not leave a directory behind to prove it was thinking about one.
            destination = pre_upgrade_destination(self.settings.data_dir, moment=int(time.time()))
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
        self,
        name: str,
        *,
        description: str | None = None,
        rule: CollectionRule | None = None,
    ) -> r.CollectionSummary:
        """Create a collection. A duplicate name is refused rather than merged.

        Raises:
            ValueError: The name is empty once normalized.
            NameInUseError: A collection of that name already exists here.
        """
        store = await self._backend.organization()
        return _collection(await store.create_collection(name, description=description, rule=rule))

    async def collection_list(self) -> r.CollectionList:
        """Every collection in this workspace."""
        store = await self._backend.organization()
        found = await store.list_collections()
        return r.CollectionList(
            count=len(found), collections=tuple(_collection(item) for item in found)
        )

    async def collection_delete(self, collection_id: str) -> r.CollectionDeleted:
        """Delete a collection. The documents in it are untouched.

        Raises:
            UnknownEntityError: No such collection in this workspace.
        """
        store = await self._backend.organization()
        await store.delete_collection(collection_id)
        return r.CollectionDeleted(collection_id=collection_id, deleted=True)

    async def collection_add(
        self, collection_id: str, document_ids: Sequence[str]
    ) -> r.CollectionMembership:
        """Add documents to a collection.

        Raises:
            UnknownEntityError: No such collection, or a document this workspace cannot see.
        """
        store = await self._backend.organization()
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
        store = await self._backend.organization()
        changed = await store.remove_from_collection(collection_id, list(document_ids))
        return r.CollectionMembership(
            collection_id=collection_id, changed=changed, document_ids=tuple(document_ids)
        )

    async def collection_rename(self, collection_id: str, name: str) -> r.CollectionSummary:
        """Rename a collection. Nothing is re-indexed and no membership moves.

        A name is a label on a row. The documents, their chunks and their embeddings are
        reached through ``collection_documents`` and the rule, neither of which mentions the
        name — so renaming cannot invalidate an embedding, and there is deliberately no
        re-index path from here for a later change to accidentally wire one up.

        Raises:
            ValueError: The name is empty once normalized.
            UnknownEntityError: No such collection in this workspace.
            NameInUseError: Another collection here already has that name.
        """
        store = await self._backend.organization()
        return _collection(await store.rename_collection(collection_id, name))

    async def collection_update(
        self, collection_id: str, *, description: str
    ) -> r.CollectionSummary:
        """Set a collection's description, leaving its membership alone.

        **``description`` has no default, and that is the whole point.** It used to default to
        ``None``, which meant every surface could reach this verb without mentioning the field
        — and ``describe_collection`` writes whatever it is given, so ``collection update <id>``
        with no arguments silently erased the description it was named for changing. A verb
        that destroys by omission is one the caller cannot be careful about. Required here, so
        a surface that forgets to pass it fails to call rather than quietly clearing.

        An empty string clears it, and clearing is therefore something asked for rather than
        something that happens. It is stored as ``None`` so that "no description" has one
        spelling on the wire instead of two that render differently.

        Raises:
            UnknownEntityError: No such collection in this workspace.
        """
        store = await self._backend.organization()
        return _collection(await store.describe_collection(collection_id, description or None))

    async def collection_rule_show(self, collection_id: str) -> r.CollectionSummary:
        """Return a collection and its current stored rule without evaluating membership."""
        store = await self._backend.organization()
        collection = await store.get_collection(collection_id)
        if collection is None:
            msg = f"no collection {collection_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)
        return _collection(collection)

    async def collection_rule_set(
        self, collection_id: str, rule: CollectionRule
    ) -> r.CollectionSummary:
        """Attach or replace a rule. Membership remains evaluated by the store on reads."""
        store = await self._backend.organization()
        return _collection(await store.set_collection_rule(collection_id, rule))

    async def collection_rule_clear(self, collection_id: str) -> r.CollectionSummary:
        """Remove a rule while preserving every manually added membership."""
        store = await self._backend.organization()
        return _collection(await store.set_collection_rule(collection_id, None))

    async def collection_counts(self, collection_id: str) -> r.CollectionCounts:
        """How many documents and chunks a collection holds, counted now.

        Both numbers are computed rather than stored. A rule-driven collection has no
        materialized membership to count, and a remembered total would keep reporting the day
        it was written.

        Raises:
            UnknownEntityError: No such collection in this workspace.
            CrossWorkspaceError: A document came back whose id was not minted here.
        """
        store = await self._backend.organization()
        collection = await store.get_collection(collection_id)
        if collection is None:
            msg = f"no collection {collection_id!r} in workspace {self.workspace!r}"
            raise UnknownEntityError(msg)

        # Counted by paging the membership rather than by a second ``COUNT`` statement.
        # ``collection_documents`` is built on the one clause that expresses membership —
        # manual rows unioned with whatever the rule selects — and a bespoke counting query
        # would be a second reader of the same rule, free to drift from it. A number that
        # disagrees with the list it claims to count is worse than a slower number.
        chunks = await self._backend.documents()
        documents = 0
        counted = 0
        offset = 0
        while True:
            page = require_owned(
                self.workspace,
                await store.collection_documents(collection_id, limit=_COUNT_PAGE, offset=offset),
            )
            if not page:
                break
            documents += len(page)
            for document in page:
                counted += await chunks.count_chunks(document.id)
            if len(page) < _COUNT_PAGE:
                break
            offset += len(page)
        return r.CollectionCounts(
            collection_id=collection_id,
            name=collection.name,
            documents=documents,
            chunks=counted,
        )

    async def collection_orphans(self, *, delete: bool = False) -> r.CollectionOrphans:
        """Live documents belonging to no collection, reported and optionally trashed.

        **Reporting is the default and deletion has to be asked for by name.** In a corpus
        where collections are optional, "in no collection" describes most of it, so a verb
        that deleted on sight would be one keystroke from emptying the workspace. The report
        is what a caller sees first, and it names what would go.

        **Deletion is into the trash**, the same soft delete ``document_delete`` performs
        without ``hard``. So this operation removes documents from the live corpus and from
        search, and every one of them can still be restored — which is the difference between
        an explicit cleanup and an irreversible one. Emptying the trash is a separate verb
        that already exists, and it stays separate.

        This is deliberately not on the HTTP API and not an MCP tool. It destroys data, and
        this project keeps that class of operation on the command line, where a person is
        present — the same rule that keeps ``reset-index``, ``backup`` and ``import`` off
        both surfaces.

        Raises:
            CrossWorkspaceError: A document came back whose id was not minted here.
        """
        store = await self._backend.organization()
        documents = await self._backend.documents()
        scope = Filter(workspace_ids=frozenset({self.workspace}))

        # The whole workspace, paged, with no cap. A capped sweep would report a count that
        # is a floor rather than a total, and ``deleted: true`` beside it would say the
        # cleanup was done when it had run out of page. This walks until the corpus ends.
        orphans: list[str] = []
        offset = 0
        while True:
            page = require_owned(
                self.workspace,
                await documents.list_documents(scope, limit=_COUNT_PAGE, offset=offset),
            )
            if not page:
                break
            for document in page:
                if not await store.collections_for(document.id):
                    orphans.append(document.id)
            if len(page) < _COUNT_PAGE:
                break
            offset += len(page)

        if delete:
            for document_id in orphans:
                await documents.soft_delete_document(document_id)
        return r.CollectionOrphans(count=len(orphans), deleted=delete, document_ids=tuple(orphans))

    async def collection_documents(
        self, collection_id: str, *, limit: int = 50, offset: int = 0
    ) -> r.DocumentList:
        """A page of a collection's documents, checked on the way out like every other read.

        Raises:
            UnknownEntityError: No such collection in this workspace.
            CrossWorkspaceError: A document came back whose id was not minted here.
        """
        store = await self._backend.organization()
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
            ValueError: The name is empty once normalized.
        """
        store = await self._backend.organization()
        return _tag(await store.ensure_tag(name, color=color))

    async def tag_list(self) -> r.TagList:
        """Every tag in this workspace."""
        store = await self._backend.organization()
        found = await store.list_tags()
        return r.TagList(count=len(found), tags=tuple(_tag(item) for item in found))

    async def tag_delete(self, tag_id: str) -> r.TagDeleted:
        """Delete a tag. Documents keep their other tags.

        Raises:
            UnknownEntityError: No such tag in this workspace.
        """
        store = await self._backend.organization()
        await store.delete_tag(tag_id)
        return r.TagDeleted(tag_id=tag_id, deleted=True)

    async def document_tag(self, document_id: str, tag_ids: Sequence[str]) -> r.DocumentTags:
        """Apply tags to a document.

        Raises:
            UnknownEntityError: No such document or tag in this workspace.
        """
        await self._require_document(document_id)
        store = await self._backend.organization()
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
        store = await self._backend.organization()
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
        store = await self._backend.organization()
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
        store = await self._backend.organization()
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
                    f"no preference judgments have been recorded at {path}. Retrieval quality "
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
                    f"{len(records)} judgment(s) were recorded and no report can be built "
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
        collection_ids: Sequence[str] = (),
    ) -> Query:
        """Build a query already carrying this workspace.

        The filter has no default workspace and cannot be built without one, so there is no
        path from here to an unscoped search.

        ``collection_ids`` are ids, already resolved from whatever the caller named by
        :meth:`_collection_scope`. Resolution happens before this rather than inside it
        because a name that names nothing must refuse the search, and a query builder that
        cannot reach the store could only drop it.
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
                collection_ids=frozenset(collection_ids),
            ),
        )

    async def _collection_scope(self, names: Sequence[str]) -> frozenset[str]:
        """Resolve collection names to ids, refusing any name that is not a collection here.

        Names rather than ids, because a name is what a person types and what an assistant
        has to hand; ids are uuids nobody quotes. Lookup goes through ``find_collection``, so
        the normalization that applied when the collection was created applies here too and a
        label typed with a trailing space finds the collection it names.

        **An unknown name is refused, never dropped.** Dropping it would leave the filter
        with no collection restriction at all, and an empty field restricts nothing — so a
        search scoped to a misspelled collection would quietly return the whole workspace,
        ranked and plausible. That is the same inversion ``resolve_filter`` returns ``None``
        to prevent, arriving one layer earlier.

        Raises:
            UnknownEntityError: A name that is not a collection in this workspace.
        """
        if not names:
            return frozenset()
        store = await self._backend.organization()
        resolved: set[str] = set()
        for name in names:
            found = await store.find_collection(name)
            if found is None:
                msg = (
                    f"no collection {name!r} in workspace {self.workspace!r}. The search is "
                    f"refused rather than run unscoped: a restriction that silently vanished "
                    f"would return every document in the workspace."
                )
                raise UnknownEntityError(msg)
            resolved.add(found.id)
        return frozenset(resolved)

    async def _require_scoped_context(self, retrieved: RetrievalResult) -> dict[str, Document]:
        """Prove every retrieved passage belongs to this workspace, before anything leaves.

        Returns the documents it resolved, which it previously discarded. Every document a
        citation can name is in here — a citation's chunk is one of these passages — so the
        answer path builds its source references from this hydration rather than reading the
        same rows again after the model has run. Re-reading them later would also have to happen
        in the ``finally`` that assembles the payload, which runs on cancellation, and a database
        read on a canceled path is a second failure mode on top of the first.
        """
        return await self._require_scoped_chunks(
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
        found = await self._scoped_documents(wanted)
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

    async def _scoped_documents(self, document_ids: Iterable[str]) -> dict[str, Document]:
        """The documents of ``document_ids`` this workspace can see, keyed by id.

        One query rather than one per document. ``document_ids`` is a field the store honors,
        so this is the scoped, trash-excluding lookup — and a ranked page routinely spans
        several documents, which made the per-document form N round trips on the hot path of
        every search and every answer.

        Returns what was found and says nothing about what was not. Two callers want different
        things from that: a passage whose document is missing is a leak and must raise, while a
        glossary entry whose document is missing is a race and must be dropped. Deciding here
        would force one of them to un-decide it.
        """
        asked = frozenset(document_ids)
        if not asked:
            return {}
        store = await self._backend.documents()
        page = await store.list_documents(
            Filter(workspace_ids=frozenset({self.workspace}), document_ids=asked),
            limit=len(asked),
        )
        # Restricted to what was asked for. A store that returns more than the filter allowed
        # is a defect its own conformance suite owns; here the question is only which of the
        # requested documents came back.
        return {document.id: document for document in page if document.id in asked}

    def _answer_payload(
        self,
        question: str,
        envelope: AnswerEnvelope,
        aside: AskAside,
        started: float,
        conversation_id: str | None,
        *,
        documents: dict[str, Document],
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
                    provenance=source_reference(documents.get(citation.document_id)),
                )
                for citation in envelope.citations
            ),
            dropped=len(envelope.dropped),
            confidence=envelope.confidence,
            expansions=aside.expansions,
            conflicts=aside.conflicts,
            confidence_band=aside.confidence_band,
            confidence_reason=aside.confidence_reason,
            explicit_definition=aside.explicit_definition,
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


def cited_definition(
    confidence: Confidence | None, expansions: Sequence[r.GlossaryExpansion]
) -> bool:
    """Whether a result may report that it is showing the definition of a term it was asked about.

    **The classification is retrieval's and is copied, never recomputed.** It is read straight
    off :attr:`~manicule.core.retrieval.Confidence.explicit_definition`, which is where the
    three conditions behind it are decided and the only place they can be decided: the question's
    shape, the entry that fired, and whether the defining passage reached the context. Deriving
    it here — from ``confidence_reason``, from the presence of an expansion, from anything at
    all — would be a second opinion, and the one that disagrees is always the one nobody reads.

    Two things can make it ``false`` that retrieval's own answer cannot.

    ``confidence is None`` is a query the router answered without consulting the corpus, and
    ``Confidence`` is *absent* on those rather than zero. There is no classification to copy
    because nothing was classified, and ``false`` is the only honest reading of a lookup that
    never happened.

    An empty ``expansions`` is the race :meth:`ApplicationService._glossary_payloads` documents:
    the entry's document became unreadable between the lookup and the render, so the provenance
    was dropped rather than shown blank. The claim goes with it. That is not a policy choice
    made here — :class:`~manicule.app.results.Glossed` refuses the combination outright, so the
    alternative is not "report it anyway" but "raise", and turning a soft delete into a failed
    search is the outcome that helper exists to avoid.

    **Withdrawing the claim does not rewrite the sentence beside it, and that is deliberate.** On
    that path ``confidence_reason`` is left saying a definition was cited, which by then is
    stale: :data:`~manicule.retrieval.confidence.DEFINITION_CITED` ends "*and the citation
    resolves to it*" and the citation has stopped resolving. The repair that suggests itself is
    to substitute :data:`~manicule.retrieval.confidence.NOTHING_RESEMBLES` here. **Do not.** It
    is false in the same state, and demonstrably so — the defining passage is still rank 1, so
    that sentence would be printed directly above the corpus's own definition of the term the
    question named. Both available sentences are wrong here, a third would be a permanent concept
    describing a window between two reads of one request, and prose this application invents is
    worse than prose it merely relays. The argument in full, with the state constructed and the
    counter-example asserted, is in ``tests/glossary/test_surfaces.py``.
    """
    return bool(confidence and confidence.explicit_definition and expansions)


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


def source_reference(document: Document | None) -> r.SourceReference | None:
    """A document's authoritative source metadata, in the shape every surface reports.

    ``None`` when the document carries no record, which is the ordinary case for a local file and
    is why the field is nullable rather than an empty object: "there is no canonical address" and
    "the canonical address is the empty string" are different claims.

    Built here, once, because four callers need it — a search hit, an answer citation, a document
    summary and the workbench — and a second copy of this mapping is the field nobody remembers
    when :class:`~manicule.core.provenance.SourceMetadata` gains one.

    **Three timestamps go in under three names and none substitutes for another.**
    ``modified_at`` is the source's, ``retrieved_at`` is the snapshot's, ``indexed_at`` is this
    installation's.

    **``snapshot_checksum`` is the local copy's digest, which is usually — and not always —
    ``content_hash``.** For every document whose stored bytes *are* the file, the column is the
    one authority for that digest and the record keeps no second copy. An enriched export is the
    exception this now has to handle: its stored bytes are the storage body *extracted from* the
    file, so ``content_hash`` digests what was indexed and the file on disk is a different,
    larger thing. Reporting the column there labels the body's digest as the snapshot's, which is
    an audit reading a citation, checksumming the file it names, and finding they disagree. The
    adapter records the snapshot's own digest, and it wins where there is one.
    """
    if document is None:
        return None
    record = document.provenance
    if record is None:
        return None
    published = record.source
    snapshot = record.snapshot
    adaptation = document.metadata.get(ENRICHED_KEY)
    # ``dict`` rather than ``Mapping``: this is a value read back out of a JSON column, and
    # ``Mapping`` is imported here only for type checking.
    declared = adaptation.get("snapshot_checksum") if isinstance(adaptation, dict) else None
    return r.SourceReference(
        title=published.title if published else "",
        canonical_uri=published.canonical_uri if published else "",
        source_id=published.source_id if published else "",
        version=published.version if published else "",
        content_type=published.content_type if published else "",
        modified_at=(
            published.modified_at.isoformat() if published and published.modified_at else None
        ),
        section_path=published.section_path if published else (),
        snapshot_path=snapshot.path if snapshot else "",
        snapshot_checksum=declared
        if isinstance(declared, str) and declared
        else document.content_hash,
        retrieved_at=(
            snapshot.retrieved_at.isoformat() if snapshot and snapshot.retrieved_at else None
        ),
        indexed_at=document.indexed_at.isoformat() if document.indexed_at else None,
        unavailable_reason=record.unavailable_reason,
    )


def _citation(citation: Citation) -> r.AnswerCitation:
    """One stored citation, in the shape every surface reports citations in.

    **No source reference, deliberately.** A stored citation is a record of what was cited, and
    its ``title`` and ``uri`` are the canonical values frozen at the moment the answer was given.
    :func:`source_reference` reports the document *now* — its current version, its current
    snapshot — so attaching one here would put live state inside a historical record and let a
    replayed conversation claim it had cited a version that did not exist yet. The rule across
    the surfaces is one sentence: a citation says what was shown, a source reference says what is
    true, and a replay only has the first.
    """
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
        provenance=source_reference(document),
    )


def _ingest_payload(report: RunReport, started: float) -> r.IngestReport:
    incomplete = report.retry_required
    bounded = report.limited and not incomplete
    reason: r.ErrorInfo | None = None
    if incomplete:
        reason = r.ErrorInfo(
            type=report.error_type or "IncompleteIngestError",
            message=report.error_message
            or (
                "a fresh source inventory is required before this snapshot can be promoted"
                if report.inventory_recovery == "reenumeration_required"
                else f"{report.snapshot_omissions} source snapshot member(s) require retry"
                if report.snapshot_omissions
                else f"{report.unrecorded} accepted document(s) left no durable record"
                if report.unrecorded
                else "retained source snapshots still require local indexing"
                if report.pending_derivation
                else report.error
            ),
            hint=(
                "Run the same ingest operation again; it will fence the stale inventory, "
                "re-enumerate from the committed watermark, and reuse validated retained bodies."
                if report.inventory_recovery == "reenumeration_required"
                else "Resolve the source acquisition failure and run the same ingest operation "
                "again; its snapshot and watermark were not published."
                if report.snapshot_omissions
                else "Run the same ingest operation again; retained source snapshots will be "
                "retried without contacting the source."
                if report.pending_derivation
                else "Run the same ingest operation again; its watermark was not advanced."
            ),
        )
    elapsed_ms = _millis(started)
    lifecycle = r.LifecycleProgress.model_validate(report.lifecycle_metadata()).model_copy(
        update={
            "rate_items_per_second": (report.discovered * 1000 / elapsed_ms if elapsed_ms else 0)
        }
    )
    return r.IngestReport(
        connector=report.connector,
        discovered=report.discovered,
        ingested=report.indexed,
        skipped=(
            report.durable_reused
            if report.durable_reused is not None
            else report.skipped_version + report.skipped_hash
        ),
        failed=(
            report.durable_failed
            if report.durable_failed is not None
            else report.by_status.get(DocumentStatus.FAILED.value, 0)
        ),
        expanded=report.expanded,
        by_status=dict(report.by_status),
        error=report.error,
        outcome="incomplete" if incomplete else "bounded" if bounded else "complete",
        enumeration_completed=report.enumeration_completed,
        watermark_advanced=report.watermark_advanced,
        snapshot_completeness=report.snapshot_completeness,
        snapshot_omissions=report.snapshot_omissions,
        snapshot_omission_reasons=dict(report.snapshot_omission_reasons),
        inventory_recovery=report.inventory_recovery,
        reconciled_deleted_items=report.reconciled_deleted_items,
        full_inventory_authority=_public_full_inventory_authority(report.full_inventory_authority),
        retry_required=incomplete,
        derivation_deferred=report.derivation_deferred,
        intentionally_bounded=bounded,
        unrecorded=report.unrecorded,
        incomplete_reason=reason,
        elapsed_ms=elapsed_ms,
        lifecycle=lifecycle,
    )


def _configured_full_inventory_authority(
    configured: ConnectorSettings,
) -> r.FullInventoryAuthority:
    """Effective public authority without constructing a credentialed live connector."""
    from manicule.connectors.config import (  # noqa: PLC0415
        CONNECTOR_NAME,
        ConfluenceConfig,
    )

    if configured.type != CONNECTOR_NAME:
        return ""
    config = ConfluenceConfig.model_validate(configured.options)
    return cast("r.FullInventoryAuthority", config.effective_full_inventory_authority.value)


def _public_full_inventory_authority(value: str) -> r.FullInventoryAuthority:
    """Closed aggregate value; durable/plugin strings never become a public data channel."""
    if value == "search":
        return "search"
    if value == "direct_current_content":
        return "direct_current_content"
    return ""


def _weights_check(plan: WeightsPlan | None) -> r.Check:
    """Turn a weights plan into the check ``doctor`` prints. Pure: no probe, no network.

    Separate from the probe so that ``init`` can ask the cache **once** and use the answer
    twice — for the note it prints and for the flag it puts on the report. Two probes would
    not merely be wasteful; they would be two answers to one question, free to disagree the
    moment somebody's first ingest finished between them.
    """
    if plan is None:
        return r.Check(
            name="models",
            state="unknown",
            detail=(
                "the embeddings extra is not installed, so there is no backend here to have "
                "weights. Install `manicule[embeddings]`."
            ),
            # The idiom this project documents (README, "Install"). `pip install` would be
            # advice for a distribution nobody installs that way — this is a checkout and uv.
            remedy="uv sync --all-extras",
        )
    if plan.error:
        return r.Check(
            name="models",
            state="failing",
            detail=f"the configured embedding artifact is invalid: {plan.error}",
            facts={
                "repo": plan.repo,
                "provider": plan.provider,
                "present": False,
                "configuration_valid": False,
            },
            remedy=plan.error,
        )
    if plan.present:
        return r.Check(
            name="models",
            state="ok",
            detail=f"{plan.repo} is on this machine; no download is pending",
            facts={
                "repo": plan.repo,
                "provider": plan.provider,
                "present": True,
                "card_present": plan.card_present,
                "revision": plan.revision,
                "ref": plan.ref,
            },
        )
    # Named per backend rather than as one line for both. `--full --mlx` fetches the parity
    # model, the ONNX export and the MLX conversion — about 3.6 GB to seed a backend that
    # loads a third of it — so the advice a Mac reader follows has to be the narrow flag, or
    # it costs them more than saying nothing would have.
    seed = f"uv run tools/prefetch_embedding_models.py --backend {plan.provider}"
    offline_env = _hub_offline_env()
    offline = os.environ.get(offline_env) if offline_env else None
    if offline:
        # The variable's **name and that it is set**, never its value. What matters here is
        # that fetching is forbidden, which the name alone says; the value is an environment
        # variable's contents, and a diagnostic that prints those is one nobody can paste into
        # an issue. It is also the specific thing this command is not allowed to do.
        return r.Check(
            name="models",
            state="failing",
            detail=(
                f"{plan.repo} is not on this machine and {offline_env} is set, which forbids "
                f"fetching it, so the first search refuses rather than waits. Seed it with "
                f"`{seed}` from a host that can reach the hub, or point embedding.model at a "
                f"local directory holding the weights."
            ),
            facts={
                "repo": plan.repo,
                "provider": plan.provider,
                "present": False,
                "card_present": plan.card_present,
                "offline_env": offline_env,
                "offline_env_set": True,
                "revision": plan.revision,
                "ref": plan.ref,
            },
            remedy=seed,
        )
    return r.Check(
        name="models",
        state="ok",
        detail=(
            f"{plan.repo} is not on this machine yet, so the first `manicule index` downloads "
            f"{plan.size} before it indexes anything. Nothing is wrong; it is a wait, once. "
            f"Run `{seed}` to take it now instead."
        ),
        facts={
            "repo": plan.repo,
            "provider": plan.provider,
            "present": False,
            "card_present": plan.card_present,
            "size": plan.size,
            "revision": plan.revision,
            "ref": plan.ref,
        },
        remedy=seed,
    )


def _hub_offline_env() -> str:
    """The hub's own offline switch, named without importing an extra that may be absent.

    Read from :mod:`manicule.embedding.runtimes.hub` rather than repeated as a literal, so the
    name ``doctor`` prints and the name the fetch actually consults cannot drift. An install
    with no embeddings extra gets an empty string, which reads as "there is no such switch
    here" — correct, because there is also no backend to switch off.
    """
    try:
        from manicule.embedding.runtimes.hub import OFFLINE_ENV  # noqa: PLC0415 - an extra
    except ImportError:
        return ""
    return OFFLINE_ENV


def _millis(started: float) -> int:
    return max(int((time.monotonic() - started) * 1000), 0)


def _severity(state: r.CheckState) -> int:
    return {"ok": 0, "degraded": 1, "failing": 2, "unknown": 1}.get(state, 1)


def _listed(names: Iterable[str]) -> str:
    """Source names in a sentence, sorted, quoted, comma separated.

    Sorted so that a diagnostic reads the same twice running, which matters when somebody is
    comparing two ``doctor`` outputs to see what changed.
    """
    return ", ".join(repr(name) for name in sorted(names))


def _ignored(message: str) -> None:
    """A progress callback for a request that cannot report any.

    Named rather than written as a lambda at the call site, because a lambda that discards its
    argument reads as an oversight and this is a statement: a ``Held`` frame is answered from a
    dictionary, so there is nothing for the server to say on the way.
    """
    del message


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


def pre_upgrade_destination(data_dir: Path, *, moment: int) -> Path:
    """Where ``upgrade`` writes the snapshot it takes before anybody installs anything.

    **A sibling of the data directory, never inside it.** The obvious place —
    ``<data_dir>/backups/`` — is refused by :func:`~manicule.storage.backup.create_backup`,
    which will not write a snapshot into the directory it is snapshotting because the copy
    would include itself. That guard is right and the caller was wrong, and the result was a
    command whose documented default form failed every time it ran
    ([#66](https://github.com/mgd43b/manicule/issues/66)): the only way to upgrade was to pass
    ``--skip-backup``, which skips the part that is dangerous to skip.

    Named from the data directory rather than fixed, so two installations on one machine do
    not write into each other's, and placed beside it rather than under ``$XDG_STATE_HOME``
    because a pre-upgrade snapshot is a copy of the corpus — not state, not cache, and not
    something to put where a cleaner might reasonably delete it. The trade is that ``doctor``
    examines ``<data_dir>`` and does not look here; the snapshot directory is created and
    verified ``0700`` by ``backup`` itself, and ``upgrade`` reports the path it used.

    Args:
        data_dir: The directory being backed up.
        moment: Unix seconds, which becomes the leaf name. Passed in rather than read here,
            so the caller — and a test — can say what it is.

    Returns:
        An absent directory, ready to be created by whoever writes into it.
    """
    return data_dir.parent / f"{data_dir.name}-backups" / f"pre-upgrade-{moment}"


def _local(path: Path | str) -> Path:
    """A user-supplied path, with ``~`` expanded.

    A module function rather than an inline call so that expansion — which reads the
    environment and the password database, never the filesystem — is not mistaken for the
    blocking I/O that has to leave the event loop.
    """
    return Path(path).expanduser()


def selected_provider(
    *,
    browser: bool,
    browser_provider: str | None,
    browser_state: Path | str | None,
    manual_cookie: bool,
    configured: BrowserProvider,
    injected: BrowserSessionProvider | None = None,
) -> BrowserProvider:
    """Which way in this login uses, decided once, from flags and then from configuration.

    **A module function taking `configured`, rather than a method reading settings**, because the
    command line has to know the answer *before* a service exists: it decides whether to prompt
    for a Cookie header, and a prompt appearing for a workspace that defaults to a browser would
    ask somebody for exactly the thing the setting exists to avoid. Two implementations of this
    rule would agree everywhere except that case.

    **The decision is here rather than spread across the branches it feeds**, because the property
    that matters is a negative one — that no path can silently become another path — and a
    negative is only checkable where the whole choice is visible. A `--browser-provider` that
    cannot start must raise; it must not arrive at the paste prompt having quietly become
    `manual_cookie`.

    Precedence is flags first, configuration second, and there is no third step. An explicit flag
    is somebody overriding the workspace for one run; the workspace default is what a bare
    `connector login` means; and a workspace that set nothing gets `manual_cookie`, which is what
    every configuration written before this setting existed already did.

    `--browser` is kept and resolves to *the configured browser* rather than to bundled Chromium.
    On a workspace that opted in to `installed_chromium` that is the only reading which is not a
    surprise, and on one that did not it lands on bundled Chromium — what the flag has always
    meant. A workspace whose default is a non-browser provider still gets a browser from it,
    because asking for a browser and being handed a paste prompt is the silent substitution this
    function exists to prevent.

    Args:
        injected: A provider object supplied by a caller, which is the test seam. Its presence
            means a browser path by definition — an object that opens a browser is the only thing
            a caller passes one for — so it selects the configured browser rather than being
            overridden by a `manual_cookie` default.
    """
    browsers = (BrowserProvider.INSTALLED_CHROMIUM, BrowserProvider.BUNDLED_CHROMIUM)
    if browser_provider is not None:
        return _named_provider(browser_provider)
    if browser_state is not None:
        return BrowserProvider.BROWSER_STATE
    if manual_cookie:
        return BrowserProvider.MANUAL_COOKIE
    if browser or injected is not None:
        return configured if configured in browsers else BrowserProvider.BUNDLED_CHROMIUM
    return configured


def _named_provider(requested: str) -> BrowserProvider:
    """One `--browser-provider` value as a member, accepting the spelling a terminal invites.

    Hyphens and underscores are both accepted because the flag reads as `installed-chromium` on
    a command line and the setting reads as `installed_chromium` in TOML, and a person who typed
    the other one has made no mistake worth a refusal. Case is folded for the same reason.

    Raises:
        ConfigError: The value names no provider. The message lists them, because a refusal for
            a mistyped enum that does not say what the options were is a refusal that costs a
            second run to learn nothing.
    """
    folded = requested.strip().lower().replace("-", "_")
    try:
        return BrowserProvider(folded)
    except ValueError:
        known = ", ".join(member.value.replace("_", "-") for member in BrowserProvider)
        msg = f"--browser-provider {requested!r} names no provider. Available: {known}."
        raise ConfigError(msg) from None


def _state_cookies(
    config: ConfluenceConfig, path: Path | str, *, allow_insecure: bool
) -> Mapping[str, SecretStr]:
    """The cookies in a Playwright state file that belong to this instance.

    Three steps, each independently testable and each refusing for its own reason: read the file
    (permissions, size), parse it (shape), filter it (origin). The last is the same function the
    browser path uses, so a state file and a live browser cannot disagree about which cookies are
    Confluence's.

    Raises:
        ConfigError: The file cannot be read or is not a state document, or it carries nothing
            for this instance. The message never includes any of the contents — the whole file
            is secret, and a diagnostic quoting the offending entry would put a session cookie
            in a terminal and a log.
    """
    from manicule.connectors.browser import (  # noqa: PLC0415 - kept beside its only use
        cookies_from_state,
        origin_cookies,
        read_state_file,
    )

    found = origin_cookies(
        cookies_from_state(read_state_file(_local(path), allow_insecure=allow_insecure)),
        base_url=config.base_url,
    )
    if not found:
        msg = (
            f"the browser state has no cookies for {config.base_url}. A state file records the "
            f"cookies of whatever was signed in when it was written, so this one is most likely "
            f"from a different site — or from a sign-in that did not reach Confluence itself. "
            f"Nothing has been stored and any previously stored session is untouched."
        )
        raise ConfigError(msg)
    return found


def _bounded_root(candidate: Path | str | None, root: Path, source: str) -> Path:
    """Where a ``--source`` conversion may walk: the connector's root, or a directory inside it.

    ``None`` is the ordinary case and means the whole root. Anything else is a **caller-supplied
    narrowing**, and narrowing is the operation that has to be bounded — a source name that could
    be paired with any path at all would make the configured root decorative and turn a connector
    into a handle for writing manifests into an unrelated tree.

    Two guards, and they answer different questions:

    *The final component may not be a symlink.* Refused outright rather than resolved, because
        the walk beneath does not follow symlinks either
        (:func:`~manicule.connectors.enriched_html._walk`). Accepting one here would make the
        entry point disagree with everything under it: a link inside the root pointing at a
        second tree would be walked as though the operator had named that tree, which they did
        not, and the manifests would land beside files the connector will never discover.

    *The fully resolved path must be the root or inside it.* Containment is checked after
        resolution rather than on the string, so ``..`` segments, an absolute path elsewhere, and
        a symlinked *parent* are all one question with one answer. Comparing the text would leave
        ``root/../../etc`` looking like a path under ``root``, which is the traversal this is for.

    A relative candidate is joined to the root, because that is what "a subdirectory of this
    source" means to somebody typing it. An absolute one is taken as given. Either way it is the
    resolved result that is checked, so the two spellings cannot get different answers.

    Raises:
        ConfigError: The candidate is a symlink, or resolves outside ``root``.
    """
    if candidate is None:
        return root
    supplied = _local(candidate)
    within = root / supplied if not supplied.is_absolute() else supplied
    if within.is_symlink():
        msg = (
            f"{str(supplied)!r} is a symbolic link. Sidecar generation does not follow symlinks "
            f"— neither does the walk beneath it, nor the connector that later reads the corpus "
            f"— so a link is refused rather than resolved. Name the directory itself."
        )
        raise ConfigError(msg)
    try:
        resolved = within.resolve()
    except OSError as exc:  # pragma: no cover - an unresolvable path is refused the same way
        msg = f"{str(supplied)!r} cannot be resolved ({exc.strerror or exc})"
        raise ConfigError(msg) from exc
    if resolved != root and root not in resolved.parents:
        msg = (
            f"{str(supplied)!r} resolves to {resolved}, which is outside {source!r}'s configured "
            f"root {root}. A source's root is what bounds every file this may write beside; a "
            f"path outside it is a different corpus, so name it as an argument on its own "
            f"instead of narrowing a source to reach it."
        )
        raise ConfigError(msg)
    return resolved


def _relative_to(path: Path, root: Path) -> str:
    """``path`` under ``root``, or its own string when it is somehow not under one.

    The fallback is not expected — the walk refuses anything outside the root — but a report is
    not worth raising over, and a full path is a true statement where a wrong relative one would
    be a misleading short answer.
    """
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:  # pragma: no cover - the walk already refuses this
        return str(path)


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


def _config_key_parts(key: str) -> list[str]:
    """Resolve a dotted CLI key, including component slots whose names contain one dot."""
    raw = key.replace('"', "").split(".")
    component_field_parts = 5
    if len(raw) >= component_field_parts and raw[:2] == ["plugins", "config"]:
        return [*raw[:2], f"{raw[2]}.{raw[3]}", *raw[4:]]
    return raw


def _validate_structural_chunker_config(
    document: Mapping[str, object], parts: Sequence[str]
) -> None:
    """Reject an accepted-but-inert component edit before the config file is written."""
    if list(parts[:3]) != ["plugins", "config", "chunker.structural"]:
        return
    from pydantic import ValidationError  # noqa: PLC0415

    from manicule.parsers.config import StructuralChunkerConfig  # noqa: PLC0415

    plugins = _as_mapping(document.get("plugins"))
    configured = _as_mapping(plugins.get("config"))
    raw = _as_mapping(configured.get("chunker.structural"))
    try:
        StructuralChunkerConfig.model_validate(raw)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        msg = f'invalid plugins.config."chunker.structural": {detail}'
        raise ConfigError(msg) from exc


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


__all__ = ["DEFAULT_SOURCE", "DEFAULT_SWEEP_BATCH", "ApplicationService", "AskAside", "hardware"]
