"""What the application service needs, stated as protocols rather than as imports.

The service is the layer both surfaces call, and it is the only layer with any behavior in
it. Writing it against protocols buys the two properties that make that worth doing:

**The service imports no database, no model runtime and no web framework.** Those arrive
through :class:`Backend`, which :mod:`manicule.app.runtime` implements over the container.

**Every guard can be checked against a component that breaks its half of the bargain.** A
store that ignores its workspace scope, a retriever that returns another tenant's chunk, an
ingest run that reports success having indexed nothing — each is a dozen lines here and
impossible against a migrated database.

The shapes deliberately mirror the components they stand for, so the production
implementations satisfy them structurally with no adapter in between.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Imported for real rather than under TYPE_CHECKING, because it is re-exported: `answering()`
# already declares the one method the answer path needs, and a second protocol of the same
# shape here would be a copy to keep in step. The module loads no provider library.
from manicule.generation.answering import SupportsAnswer as Answering

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    from manicule.app.results import ApiKeySummary, Check
    from manicule.config.settings import Settings
    from manicule.core.acquisition import AcquisitionRun
    from manicule.core.content import Chunk, Document, DocumentStatus
    from manicule.core.embedding import IndexFingerprints
    from manicule.core.fingerprints import GlossaryFingerprint
    from manicule.core.organization import Collection as DocumentCollection
    from manicule.core.organization import CollectionRule, Restoration, Tag, TrashEntry
    from manicule.core.protocols import Connector
    from manicule.core.rebuild import RebuildCheckpoint, RebuildEstimate
    from manicule.core.retrieval import Filter, Query
    from manicule.core.source_lifecycle import LifecycleOutcome, LifecyclePlan
    from manicule.generation.history import Turn
    from manicule.generation.ports import (
        ConversationRecord,
        Feedback,
        FeedbackReason,
        SharedTurn,
    )
    from manicule.generation.sharing import ShareLink
    from manicule.ingest.pipeline import RunReport, Watching
    from manicule.ingest.reembed import ReembedPlan, ReembedRecovery, ReembedRun
    from manicule.ingest.reindex import GlossarySweep, ReindexReport, StaleSweep
    from manicule.plugins.registry import Discovery
    from manicule.retrieval.retriever import RetrievalResult


@runtime_checkable
class DocumentSurface(Protocol):
    """The reads and writes the service performs against the relational store.

    A subset of what :class:`~manicule.storage.docstore.SqliteDocStore` offers, chosen so the
    service cannot reach past what a surface legitimately does — it lists, reads, deletes and
    counts, and it cannot write a chunk or advance a watermark.
    """

    @property
    def workspace_id(self) -> str:
        """Which tenant this handle serves. Never a parameter, so it cannot be forgotten."""
        ...

    async def get_document(self, document_id: str) -> Document | None: ...

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol it narrows
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]: ...

    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        glossary_fp_other_than: str | None = None,
        glossary_fp_unrecorded: bool = False,
    ) -> int:
        """How many live documents match.

        ``glossary_fp_other_than`` is the one lineage predicate this surface can reach, and it
        is here because ``doctor`` has to be able to say "this corpus is serving definitions the
        installed detector did not produce" without being handed the ability to repair anything.
        It is a count over an indexed column: no glossary text is read to answer it.

        ``glossary_fp_unrecorded`` narrows it to the documents nothing has ever versioned, which
        is the one-time migration an index predating the column needs and a different thing to
        tell somebody than "your detector moved".
        """
        ...

    async def count_chunks(self, document_id: str | None = None) -> int: ...

    async def delete_document(self, document_id: str) -> None: ...

    async def soft_delete_document(self, document_id: str) -> None: ...

    async def document_statistics(self) -> Mapping[str, Mapping[str, int]]:
        """Document counts grouped by ``source``, ``media_type`` and ``status``.

        Three aggregates rather than a listing the surface counts itself: a page of documents
        answers a different question, and summing one would report the page rather than the
        corpus while looking exactly like a total.
        """
        ...

    async def index_fingerprints(self) -> IndexFingerprints: ...

    async def connector_metadata(self, connector: str) -> Mapping[str, object]: ...


@runtime_checkable
class Retrieving(Protocol):
    """One workspace's retrieval, end to end.

    :class:`~manicule.retrieval.retriever.Retriever` satisfies this. The service holds the
    narrower shape so a test can drive ``search`` and ``ask`` without assembling a pipeline.
    """

    async def retrieve(self, query: Query) -> RetrievalResult: ...


@runtime_checkable
class Ingesting(Protocol):
    """The ingest operations a surface can start."""

    async def index_path(
        self,
        path: Path,
        *,
        name: str,
        limit: int | None = None,
        force: bool = False,
        watching: Watching | None = None,
    ) -> RunReport:
        """Ingest a file or a directory as the source called ``name``."""
        ...

    async def sync(
        self,
        connector: str,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
        acquire_only: bool = False,
    ) -> RunReport:
        """Run one configured connector.

        ``watching`` is called with one sentence per document reaching a terminal outcome, so a
        caller streaming to a person can show a long sync moving. It is called from inside the
        pipeline and must neither block nor raise.
        """
        ...

    async def connector(self, name: str) -> Connector:
        """The constructed connector for a configured source, without running it.

        The **same object** :meth:`sync` runs, from the same container, built by the same factory
        over the same validated configuration. That is the whole reason this is on the port
        rather than being a settings lookup in the service: sidecar generation needs a
        filesystem source's resolved root and its ``enriched_profiles``, and reading those out of
        ``settings.connectors[name].options`` would be a *second* interpretation of a profile.
        Two interpretations of one profile is precisely the defect — a conversion written under
        one reading and a sync performed under another put manifests on disk for pages the
        connector then declines to read, and neither report mentions the other.

        Raises:
            UnknownComponentError: Configuration has no connector by that name. Surfaces
                translate it; the container raises it.
        """
        ...

    async def snapshot_status(self, connector: str) -> tuple[AcquisitionRun, bool] | None: ...

    async def snapshot_verify(self, run_id: str) -> tuple[AcquisitionRun, bool] | None: ...

    async def rebuild_plan(self, snapshot_run_id: str) -> RebuildEstimate: ...

    async def rebuild_run(self, snapshot_run_id: str, owner: str) -> RebuildCheckpoint: ...

    async def rebuild_status(self, generation_id: str) -> RebuildCheckpoint | None: ...

    async def reembed_plan(self) -> tuple[ReembedPlan, str, int]: ...

    async def reembed_start(self, run_id: str, owner_token: str) -> ReembedRun: ...

    async def reembed_resume(self, run_id: str, owner_token: str) -> ReembedRun: ...

    async def reembed_status(self, run_id: str) -> ReembedRun | None: ...

    async def reembed_abandon(self, run_id: str, owner_token: str) -> ReembedRun: ...

    async def reembed_cleanup(self, run_id: str) -> bool: ...

    async def reembed_recover_pending(self) -> ReembedRecovery: ...

    async def reindex(self, document_id: str) -> ReindexReport:
        """Re-parse one document from its retained bytes. No network."""
        ...

    async def reparse_stale(self, *, batch: int, dry_run: bool = False) -> StaleSweep:
        """Re-parse every document whose recorded parse lineage is no longer installed.

        The corpus-wide end of :meth:`reindex`, and it is on the port rather than assembled in
        the service for one reason: the selection is a query over lineage columns that
        :class:`DocumentSurface` deliberately cannot reach, and the pipeline it runs through
        has to be the *same* object a sync would use so that the two share one embedding lock.
        A service that built either for itself would be a second answer to a question the
        runtime already answers.

        Which fingerprints count as current is decided here rather than passed in. A partial
        set makes every document its parser produced look stale, which is a repair that cannot
        end, so no surface is given the chance to supply one.
        """
        ...

    async def glossary_fingerprint(self) -> GlossaryFingerprint:
        """What the installed detector would produce under this configuration.

        On this port rather than derived by the service, and it is the one accessor that keeps
        the whole feature coherent: ``status`` reports it, ``doctor`` counts the documents that
        disagree with it, :meth:`redetect_stale_glossary` repairs them, and the pipeline stamps
        it. Four readers of one fact, and any two of them computing it separately would
        disagree the first time a middleware chain was read a second way — which would report a
        whole corpus stale and repair it into a state ``doctor`` still called stale.

        **It builds nothing.** The answer is a digest of two source files and one configuration
        flag, so a health check can ask it without constructing an embedder — which is the whole
        difference between a diagnostic and something that needs the system working.
        """
        ...

    async def redetect_stale_glossary(self, *, batch: int, dry_run: bool = False) -> GlossarySweep:
        """Recompute the glossary of every document the installed detector did not produce.

        **The cheapest repair on this port, and separate from :meth:`reparse_stale` so that it
        can be.** It reads stored chunk text and writes rows: no connector, no retained bytes,
        no parser, no embedder, no vector. Detection is a stage of its own with rules that move
        independently of all three of those, so a corrected detector has to be able to reach a
        corpus without charging it a re-parse — which is what folding this into the parse sweep
        would have done.

        Which fingerprint counts as current is decided here for the same reason it is there:
        the answer is a fact about the installed build, and a surface allowed to supply one
        could stamp a corpus with a detector that never ran over it.

        Raises:
            PolicyError: Detection is switched off, so there is no detector to be current with.
        """
        ...

    async def import_archive(self, path: Path, *, force: bool = False) -> RunReport:
        """Ingest an exported corpus archive.

        The archive carries retained source bytes and document metadata, never chunks or
        vectors, so the importing installation produces both with **its own** fingerprints.
        Copying chunks across machines would move an index built by another chunker and
        another embedder into a store whose fingerprint says otherwise.
        """
        ...


@runtime_checkable
class Maintenance(Protocol):
    """Whole-installation operations: the ones that touch files rather than rows."""

    async def schema_revision(self) -> str | None: ...

    async def backup(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> Mapping[str, object]:
        """Snapshot the installation into ``target``. Returns the manifest.

        A group- or world-readable ``target`` is refused: a snapshot is a verbatim second
        copy of the corpus, retained source bytes included, and the flag is on the port
        rather than buried in storage because refusing is the default and consenting is a
        decision an operator makes out loud.
        """
        ...

    async def restore(self, source: Path, *, force: bool = False) -> Mapping[str, object]: ...

    async def reset_index(self) -> tuple[int, int, bool]:
        """Reset derived state; return documents affected, chunks removed, vectors removed."""
        ...

    async def plan_reset_derived(self) -> LifecyclePlan: ...

    async def reset_derived(self) -> LifecycleOutcome: ...

    async def plan_derived_generation_cleanup(self) -> LifecyclePlan: ...

    async def cleanup_derived_generations(self) -> LifecycleOutcome: ...

    async def plan_source_history_release(self, cutoff: datetime) -> LifecyclePlan: ...

    async def release_source_history(self, cutoff: datetime) -> LifecycleOutcome: ...

    async def plan_snapshot_deletion(self, run_id: str) -> LifecyclePlan: ...

    async def delete_snapshot(self, run_id: str, *, confirmation: str) -> LifecycleOutcome: ...

    async def export_corpus(
        self, target: Path, *, allow_insecure_target: bool = False
    ) -> tuple[int, int]:
        """Write a portable archive of this workspace's retained bytes and metadata.

        Returns the number of documents written and the number of bytes they occupy.

        A group- or world-readable ``target`` is refused on the same terms as :meth:`backup`,
        and for a stronger reason: an archive is written *in order to be carried somewhere*,
        so it is the copy most likely to be read by somebody the corpus was never shared with.
        """
        ...

    async def workspaces(self) -> Sequence[tuple[str, str, str]]:
        """Every workspace this installation knows about: ``(id, name, mode)``.

        Names and modes only. A cross-workspace *listing* is configuration; a cross-workspace
        *read* is a tenancy breach, and this handle cannot perform one.
        """
        ...


@runtime_checkable
class Organizing(Protocol):
    """Collections, tags and the trash, for one workspace.

    Separate from :class:`DocumentSurface` rather than folded into it, because the two are
    satisfied by the same object and required by different callers: retrieval needs a store
    that lists documents and has no business needing one that can rename a tag.
    """

    async def create_collection(
        self, name: str, *, description: str | None = None, rule: CollectionRule | None = None
    ) -> DocumentCollection: ...

    async def list_collections(self) -> Sequence[DocumentCollection]: ...

    async def get_collection(self, collection_id: str) -> DocumentCollection | None: ...

    async def find_collection(self, name: str) -> DocumentCollection | None: ...

    async def rename_collection(self, collection_id: str, name: str) -> DocumentCollection: ...

    async def describe_collection(
        self, collection_id: str, description: str | None
    ) -> DocumentCollection: ...

    async def collections_for(self, document_id: str) -> Sequence[DocumentCollection]: ...

    async def delete_collection(self, collection_id: str) -> None: ...

    async def add_to_collection(self, collection_id: str, document_ids: Sequence[str]) -> int: ...

    async def remove_from_collection(
        self, collection_id: str, document_ids: Sequence[str]
    ) -> int: ...

    async def collection_documents(
        self, collection_id: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]: ...

    async def ensure_tag(self, name: str, *, color: str | None = None) -> Tag: ...

    async def list_tags(self) -> Sequence[Tag]: ...

    async def delete_tag(self, tag_id: str) -> None: ...

    async def tag_document(self, document_id: str, tag_ids: Sequence[str]) -> int: ...

    async def untag_document(self, document_id: str, tag_ids: Sequence[str]) -> int: ...

    async def tags_for(self, document_id: str) -> Sequence[Tag]: ...

    async def list_trash(
        self, *, grace_s: float, limit: int = 100, offset: int = 0
    ) -> Sequence[TrashEntry]: ...

    async def restore_document(self, document_id: str) -> Restoration: ...


@runtime_checkable
class Conversing(Protocol):
    """Conversations, their turns, and the share links over them.

    Every method is scoped to the handle's workspace, and the sharing ones are deliberately
    the concrete store's shapes rather than a convenience layer: ``create_share`` takes the
    whole minted link and the ceiling, so nothing here can install a link on a conversation it
    was not minted for or outlive ``security.sharing.link_ttl_s``.
    """

    async def create_conversation(
        self, *, user_id: str | None = None, title: str | None = None
    ) -> str: ...

    async def list_conversations(
        self, *, limit: int = 50, offset: int = 0
    ) -> Sequence[ConversationRecord]: ...

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None: ...

    async def rename_conversation(self, conversation_id: str, title: str) -> bool: ...

    async def soft_delete_conversation(self, conversation_id: str) -> bool: ...

    async def history(self, conversation_id: str, *, limit: int = 20) -> Sequence[Turn]: ...

    async def record_feedback(
        self,
        message_id: str,
        *,
        feedback: Feedback,
        reason: FeedbackReason | None = None,
        comment: str = "",
    ) -> bool: ...

    async def create_share(self, link: ShareLink, *, maximum_ttl_s: int) -> bool: ...

    async def revoke_share(self, conversation_id: str) -> bool: ...

    async def shared_conversation(
        self, token_hash: str, *, now: datetime, sharing_enabled: bool
    ) -> Sequence[SharedTurn]:
        """The turns a live token names, already projected for an anonymous reader.

        The projection happens in the implementation, not here and not in a surface. A
        redaction a caller has to remember to apply is one a caller forgets.
        """
        ...


@runtime_checkable
class Telemetry(Protocol):
    """Query logs and the audit trail, for one workspace.

    Both are written by the service and read by the admin surface. Neither is written by a
    surface: a record only one surface produces describes only that surface's traffic, and an
    audit trail with holes in it is worse than none because the holes are invisible.
    """

    async def record_query(
        self,
        query: str,
        *,
        profile: str,
        chunk_ids: Sequence[str],
        confidence: float | None,
        elapsed_ms: int,
    ) -> str:
        """Record one retrieval and return the row's id."""
        ...

    async def query_logs(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        """A page of retrieval telemetry, newest first, and the total row count."""
        ...

    async def record_audit(
        self,
        event_type: str,
        *,
        details: Mapping[str, object],
        actor: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Record one security-relevant event."""
        ...

    async def audit_logs(
        self, *, limit: int = 50, offset: int = 0, event_type: str | None = None
    ) -> tuple[Sequence[Mapping[str, object]], int]:
        """A page of the audit trail, newest first, and the total row count."""
        ...


@runtime_checkable
class Keys(Protocol):
    """API keys for one workspace.

    Workspace-scoped like everything else, and for the same reason: a key is an identity, and
    an identity that can be minted in one tenant and used in another is not isolation.
    """

    async def issue(
        self, name: str, *, role: str, expires_days: int | None = None
    ) -> tuple[ApiKeySummary, str]:
        """Mint a key. Returns its record and the secret, which exists only here.

        The secret is returned once and never stored — only a digest is. A lost key is
        reissued rather than recovered, which is the property that makes a leaked backup not
        also a leaked credential.
        """
        ...

    async def list_keys(self) -> Sequence[ApiKeySummary]: ...

    async def revoke(self, name_or_id: str) -> ApiKeySummary:
        """Revoke a key by name or id. Immediate."""
        ...

    async def verify(self, secret: str) -> ApiKeySummary | None:
        """Which key this secret is, or ``None`` if it is not a usable one.

        ``None`` covers unknown, revoked, expired and belonging-to-another-workspace, and
        deliberately does not say which. Telling a caller that the key they presented is
        merely *expired* confirms it was once real, which is a fact worth having if you are
        collecting them.

        The comparison is over a digest, so nothing here can be timed into a byte at a time.
        """
        ...


class Backend(Protocol):
    """Everything the service is given, built lazily and never by the service itself.

    Each accessor is async and may be expensive: ``retriever`` constructs an embedder,
    ``answerer`` loads a provider library. ``doctor`` and ``stats`` must not pay for either,
    which is why they are separate calls rather than fields — a field would be built by
    whoever assembled the backend, for every command.
    """

    @property
    def settings(self) -> Settings: ...

    @property
    def workspace(self) -> str: ...

    @property
    def discovery(self) -> Discovery | None:
        """What plugin discovery found, or ``None`` if it has not run."""
        ...

    async def documents(self) -> DocumentSurface: ...

    async def retriever(self) -> Retrieving: ...

    async def answerer(self) -> Answering: ...

    async def ingestion(self) -> Ingesting: ...

    async def maintenance(self) -> Maintenance: ...

    async def organization(self) -> Organizing: ...

    async def conversations(self) -> Conversing: ...

    async def telemetry(self) -> Telemetry: ...

    async def keys(self) -> Keys: ...

    async def component_checks(self) -> Sequence[Check]:
        """Health of whatever is already constructed, without constructing anything else."""
        ...


__all__ = [
    "Answering",
    "Backend",
    "Conversing",
    "DocumentSurface",
    "Ingesting",
    "Keys",
    "Maintenance",
    "Organizing",
    "Retrieving",
    "Telemetry",
]
