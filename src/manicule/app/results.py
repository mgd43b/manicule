"""The result vocabulary, which is also the ``--json`` contract.

Every operation returns one of the payload models here, and both surfaces wrap it in the same
:class:`Envelope`. That is the whole reason the models exist: a shape defined once and dumped
by one function cannot drift between the command line and the MCP tool that runs the same
service call, and a consumer that parses one has parsed both.

The envelope carries four things before any payload:

``op``
    The operation's name, identical to the MCP tool's name. A log line, a shell pipeline and a
    tool call all name the same operation the same way.

``ok``
    Whether ``data`` or ``error`` is present. Exactly one of the two always is.

``workspace``
    Which tenant the operation ran in. Present on **every** envelope, including failures,
    because identity here is workspace-scoped and an answer whose scope is invisible cannot be
    audited.

``version``
    The contract version, which is manicule's own. A consumer that has to branch on shape has
    something to branch on that is not a guess about field presence.

``docs/surfaces.md`` is the reference; this module is the definition it describes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.core.version import CORE_VERSION

CONTRACT_VERSION = CORE_VERSION
"""What ``version`` reports. The surfaces version with the core, not separately."""


class Payload(BaseModel):
    """Base for every operation's result.

    Frozen and closed. A payload that accepted extras would let a field appear in output
    without appearing in this file, and the contract would then be whatever the code happened
    to do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorInfo(Payload):
    """What went wrong, in the shape a caller can act on."""

    type: str = Field(description="The exception class name, e.g. ``PolicyError``.")
    message: str
    hint: str = Field(
        default="",
        description="What to do about it, when there is something specific to say.",
    )


class Envelope(BaseModel):
    """One operation's outcome, as it is serialised by both surfaces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = CONTRACT_VERSION
    op: str = Field(min_length=1)
    ok: bool
    workspace: str = Field(min_length=1)
    data: dict[str, JsonValue] | None = None
    error: ErrorInfo | None = None

    def as_json(self) -> dict[str, Any]:
        """The envelope as plain JSON-safe data.

        ``exclude_none`` is deliberately **off**: a consumer checking ``ok`` then reading
        ``data`` should find the other key present and null rather than absent, because
        "absent" and "null" are the same thing in a shell pipeline and different things in a
        typed client.
        """
        return self.model_dump(mode="json")


def succeeded(op: str, workspace: str, payload: Payload) -> Envelope:
    """Wrap a payload. The only way a successful result is built."""
    return Envelope(op=op, ok=True, workspace=workspace, data=payload.model_dump(mode="json"))


def failed(op: str, workspace: str, error: ErrorInfo) -> Envelope:
    """Wrap a failure. The only way a failed result is built."""
    return Envelope(op=op, ok=False, workspace=workspace, error=error)


type CheckState = Literal["ok", "degraded", "failing", "unknown"]
"""How a diagnostic came out.

``unknown`` is not a fourth severity — it is the honest answer for a check that could
not run, and is deliberately distinct from ``ok``, which claims something was measured.

Declared here, above every payload that uses it, rather than beside :class:`Check`. Pydantic
resolves an annotation against the module namespace as the model class is built, so a payload
defined earlier in the file than its own type alias fails at import — loudly, but at import,
which is a poor place to discover an ordering rule.
"""


# --- retrieval and answers -----------------------------------------------------------------


class Anchored(Payload):
    """A location in a document, as a citation or a search hit reports it."""

    document_id: str
    chunk_id: str
    uri: str
    title: str
    heading_path: tuple[str, ...] = ()
    kind: str = ""
    anchor: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="The anchor, dumped as it is stored. Never reformatted for display: a "
        "citation's value is that it resolves, and a prettied location is one nobody can "
        "resolve back.",
    )


class SearchHit(Anchored):
    """One ranked passage."""

    score: float
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score per pipeline stage, in the order the stages ran. Kept because "
        "'reranking helped' is only checkable while the pre-rerank score survives.",
    )
    text: str
    token_count: int = 0


class SearchResult(Payload):
    """What ``search`` produced."""

    query: str
    profile: str
    count: int = Field(ge=0)
    hits: tuple[SearchHit, ...] = ()
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_reason: str = ""
    route: str = ""
    cached: bool = False
    truncated: bool = False
    elapsed_ms: int = Field(default=0, ge=0)


class AnswerCitation(Anchored):
    """A citation the answer carries, after verification."""

    slot: int = Field(ge=1)
    quote: str = ""
    verification: str = ""


class AnswerResultPayload(Payload):
    """What ``ask`` produced.

    ``confidence`` is absent rather than zero when the corpus was not consulted, and
    ``ungrounded`` means "the context was non-empty and nothing survived verification". The
    two are separate fields because they are separate claims.
    """

    question: str
    text: str
    citations: tuple[AnswerCitation, ...] = ()
    dropped: int = Field(default=0, ge=0)
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_reason: str = ""
    corpus_consulted: bool = True
    ungrounded: bool = False
    context_truncated: bool = False
    redacted: bool = False
    finish_reason: str | None = None
    error: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    model: str = ""
    elapsed_ms: int = Field(default=0, ge=0)


# --- documents -----------------------------------------------------------------------------


class DocumentSummary(Payload):
    """One document, without its text."""

    id: str
    source: str
    source_id: str
    uri: str
    title: str = ""
    media_type: str = ""
    status: str = ""
    status_detail: str | None = None
    failed_stage: str | None = None
    content_hash: str = ""
    chunk_count: int | None = Field(default=None, ge=0)


class DocumentList(Payload):
    """A page of documents."""

    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    documents: tuple[DocumentSummary, ...] = ()


class DocumentChunk(Payload):
    """One stored chunk, as ``document_get`` returns it."""

    id: str
    position: int = Field(ge=0)
    kind: str
    heading_path: tuple[str, ...] = ()
    token_count: int = Field(ge=0)
    text: str
    anchor: dict[str, JsonValue] = Field(default_factory=dict)


class DocumentDetail(Payload):
    """One document, optionally with its chunks."""

    document: DocumentSummary
    chunks: tuple[DocumentChunk, ...] = ()


class DocumentDeleted(Payload):
    """The outcome of a delete."""

    document_id: str
    deleted: bool
    mode: Literal["soft", "hard"]


class DocumentReindexed(Payload):
    """The outcome of a reindex of one document."""

    document_id: str
    status: str
    chunks: int = Field(ge=0)
    detail: str = ""


# --- conversations -------------------------------------------------------------------------


class ConversationSummary(Payload):
    """One conversation, without its turns.

    ``share_token`` is absent by construction. A listing that carried it would hand a bearer
    capability to every reader of the listing, and put it into every log and cache in front of
    the surface that returned it — which is the whole reason the stored form is a hash.
    """

    id: str
    title: str | None = None
    shared: bool = False
    shared_at: str | None = None
    share_expires_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    messages: int = Field(default=0, ge=0)


class ConversationList(Payload):
    """A page of conversations."""

    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    conversations: tuple[ConversationSummary, ...] = ()


class ConversationTurn(Payload):
    """One turn as its **owner** reads it: full citations, passage text and all."""

    role: str
    content: str
    citations: tuple[AnswerCitation, ...] = ()


class ConversationMessages(Payload):
    """A conversation's turns, oldest first."""

    conversation_id: str
    count: int = Field(ge=0)
    turns: tuple[ConversationTurn, ...] = ()


class ConversationDeleted(Payload):
    """The outcome of deleting a conversation. Deleting also revokes any share link."""

    conversation_id: str
    deleted: bool
    share_revoked: bool = Field(
        default=True,
        description="Always true on a successful delete. A soft delete that left a public "
        "link resolving is a delete that did not delete.",
    )


class ConversationRenamed(Payload):
    """The outcome of retitling a conversation."""

    conversation_id: str
    title: str


class ShareCreated(Payload):
    """A minted share link. The token appears here and nowhere else, ever."""

    conversation_id: str
    token: str = Field(
        description="Shown once. Only its SHA-256 digest is stored, so a lost link is "
        "re-minted rather than recovered — and re-minting invalidates the previous one."
    )
    path: str = Field(description="Where the link resolves, relative to the server's root.")
    expires_at: str
    shared_at: str


class ShareRevoked(Payload):
    """The outcome of revoking a link. Revocation clears the stored hash."""

    conversation_id: str
    revoked: bool


class SharedCitationLabel(Payload):
    """What an anonymous viewer is told about one citation.

    A **different shape** from :class:`AnswerCitation`, not a blanked-out one. There is
    nowhere here to put a document id, a chunk id, a URI, an anchor or the quoted passage, so
    a route cannot leak one by forgetting to clear a field.
    """

    slot: int = Field(ge=1)
    title: str
    heading_path: tuple[str, ...] = ()
    location: str = ""
    verification: str = ""


class SharedTurnPayload(Payload):
    """One turn of a shared conversation, as an anonymous viewer receives it."""

    role: str
    content: str
    citations: tuple[SharedCitationLabel, ...] = ()


class SharedConversation(Payload):
    """A shared conversation, for a reader with no workspace membership.

    Carries **no conversation id**. Handing one back would let a holder of the link address
    the conversation by id elsewhere, which is the two-step the single-statement resolution in
    :meth:`~manicule.storage.conversations.SqliteConversationStore.shared_conversation`
    exists to replace.
    """

    count: int = Field(ge=0)
    turns: tuple[SharedTurnPayload, ...] = ()


class FeedbackRecorded(Payload):
    """The outcome of rating an answer."""

    message_id: str
    recorded: bool
    feedback: str


# --- collections and tags ------------------------------------------------------------------


class CollectionSummary(Payload):
    """One collection. ``rule`` is present when membership is evaluated rather than stored."""

    id: str
    name: str
    description: str | None = None
    rule: dict[str, JsonValue] | None = None
    created_at: str = ""


class CollectionList(Payload):
    """Every collection in this workspace."""

    count: int = Field(ge=0)
    collections: tuple[CollectionSummary, ...] = ()


class CollectionMembership(Payload):
    """The outcome of adding documents to a collection, or removing them."""

    collection_id: str
    changed: int = Field(ge=0)
    document_ids: tuple[str, ...] = ()


class CollectionDeleted(Payload):
    """The outcome of deleting a collection. The documents in it are untouched."""

    collection_id: str
    deleted: bool


class TagSummary(Payload):
    """One tag."""

    id: str
    name: str
    color: str | None = None


class TagList(Payload):
    """Every tag in this workspace."""

    count: int = Field(ge=0)
    tags: tuple[TagSummary, ...] = ()


class TagDeleted(Payload):
    """The outcome of deleting a tag. Documents keep their other tags."""

    tag_id: str
    deleted: bool


class DocumentTags(Payload):
    """A document's tags after an application or removal."""

    document_id: str
    changed: int = Field(ge=0)
    tags: tuple[TagSummary, ...] = ()


# --- the trash -----------------------------------------------------------------------------


class TrashedDocument(Payload):
    """One soft-deleted document, and what restoring it would cost."""

    document: DocumentSummary
    deleted_at: str
    purged: bool = False
    restorable_until: str | None = None
    free_restore: bool = True


class TrashList(Payload):
    """A page of the trash, longest-deleted first — the order the sweep takes them in."""

    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    documents: tuple[TrashedDocument, ...] = ()


class DocumentRestored(Payload):
    """What restoring a document achieved, and what is still needed."""

    document_id: str
    restored: bool
    needs_reparse: bool = False
    reason: str


# --- telemetry and the audit trail ---------------------------------------------------------


class QueryLogEntry(Payload):
    """One recorded retrieval."""

    id: str
    query: str
    profile: str = ""
    chunks: int = Field(default=0, ge=0)
    confidence: float | None = None
    elapsed_ms: int | None = None
    created_at: str = ""


class QueryLogPage(Payload):
    """A page of retrieval telemetry, newest first."""

    total: int = Field(ge=0)
    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    entries: tuple[QueryLogEntry, ...] = ()


class AuditEntry(Payload):
    """One security-relevant event."""

    id: str
    event_type: str
    actor: str | None = None
    ip_address: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: str = ""


class AuditPage(Payload):
    """A page of the audit trail, newest first.

    ``enabled`` is on the payload rather than implied by an empty list. "Nothing happened" and
    "nothing was recorded because auditing is off" are different answers, and an operator
    reading an empty audit log needs to know which one they have.
    """

    enabled: bool
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    entries: tuple[AuditEntry, ...] = ()


class SearchQuality(Payload):
    """What the evaluation harness has actually recorded.

    Deliberately a report on :mod:`manicule.evaluation`'s own store rather than a second
    scoring path. ``is_evidence`` is false when the query set is an example one, and the
    ``caveat`` says so in words — an example query set is an illustration, and presenting one
    as a measurement is the failure the whole harness exists to prevent.
    """

    available: bool = Field(description="Whether any judgements have been recorded at all.")
    is_evidence: bool = False
    caveat: str = ""
    path: str = ""
    left_label: str = ""
    right_label: str = ""
    query_set: str = ""
    records: int = Field(default=0, ge=0)
    judged: int = Field(default=0, ge=0)
    report: str = Field(default="", description="The harness's own rendering, verbatim.")


class PluginHealth(Payload):
    """One installed plugin and the health of what it registered."""

    name: str
    version: str = ""
    enabled: bool = True
    components: int = Field(default=0, ge=0)
    state: CheckState = "unknown"
    detail: str = ""


class PluginHealthReport(Payload):
    """Plugin health, for the admin surface."""

    count: int = Field(ge=0)
    plugins: tuple[PluginHealth, ...] = ()
    disabled: tuple[str, ...] = ()


# --- the workbench -------------------------------------------------------------------------


class WorkbenchBlock(Payload):
    """One chunk of a document, with the anchor that locates it."""

    id: str
    position: int = Field(ge=0)
    kind: str
    heading_path: tuple[str, ...] = ()
    token_count: int = Field(ge=0)
    text: str
    anchor: dict[str, JsonValue] = Field(default_factory=dict)


class Workbench(Payload):
    """A document as it was chunked, for inspecting what retrieval actually sees.

    Read-only, and one document at a time. It exists so a person can look at the units the
    index is built from — which is the only way to tell a chunking problem from a retrieval
    one — and it invents nothing: the blocks are the stored chunks.
    """

    document: DocumentSummary
    count: int = Field(ge=0)
    tokens: int = Field(default=0, ge=0)
    blocks: tuple[WorkbenchBlock, ...] = ()


# --- identity ------------------------------------------------------------------------------


class Identity(Payload):
    """Who the caller is, as the surface that authenticated them sees it."""

    authenticated: bool
    mode: str = Field(description="The configured auth mode: ``none``, ``api_key`` or ``oauth``.")
    role: str = ""
    key_id: str = ""
    key_name: str = ""
    workspace: str = ""


class AuthProviders(Payload):
    """The identity providers this installation is configured for.

    Names and types only, never a client secret. Empty when ``security.auth.mode`` is not
    ``oauth``, which is the honest answer rather than a list nothing would accept.
    """

    mode: str
    count: int = Field(ge=0)
    providers: tuple[str, ...] = ()
    detail: str = ""


# --- ingest --------------------------------------------------------------------------------


class IngestReport(Payload):
    """What one ingest run did. Shared by ``index_path``, ``connector_sync`` and ``import``.

    ``by_status`` is the run's own counter table rather than a summary of it, because a
    document that ended ``no_extractable_text`` is neither an ingest nor a failure and
    collapsing the two would hide exactly the outcome that needs looking at.
    """

    connector: str
    discovered: int = Field(default=0, ge=0)
    ingested: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    expanded: int = Field(default=0, ge=0)
    by_status: dict[str, int] = Field(default_factory=dict)
    error: str = ""
    elapsed_ms: int = Field(default=0, ge=0)


# --- state, statistics and diagnosis -------------------------------------------------------


class IndexStatus(Payload):
    """What is in the index, and whether it is coherent.

    Deliberately thin. Operations (#14) owns dashboards, alerting and history; this reports
    what the store already knows, at the moment it is asked.
    """

    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    by_status: dict[str, int] = Field(default_factory=dict)
    embedding: str = Field(
        default="",
        description="What the embedder is, in one line. For reading; ``embed_fingerprint`` "
        "is what to compare.",
    )
    chunking: str = Field(default="", description="What the chunker is, in one line.")
    embed_fingerprint: str | None = Field(
        default=None,
        description="The index's committed embedding identity, canonically. ``None`` means "
        "the index has committed to nothing and will accept whatever the first ingest brings.",
    )
    chunk_fingerprint: str | None = None
    schema_revision: str | None = None
    data_dir: str = ""


class Stats(Payload):
    """Counts a person would ask for. Thin for the same reason as :class:`IndexStatus`."""

    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    by_source: dict[str, int] = Field(default_factory=dict)
    by_media_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


class Check(Payload):
    """One diagnostic."""

    name: str
    state: CheckState
    detail: str = ""


class Diagnosis(Payload):
    """Everything ``doctor`` looked at."""

    state: CheckState
    checks: tuple[Check, ...] = ()


# --- connectors ----------------------------------------------------------------------------


class ConnectorSummary(Payload):
    """One configured source."""

    name: str
    type: str
    enabled: bool = True
    schedule_s: int | None = None
    installed: bool = True
    last_synced_at: str | None = None
    status: str = ""
    documents: int | None = Field(default=None, ge=0)


class ConnectorList(Payload):
    """Every configured source."""

    count: int = Field(ge=0)
    connectors: tuple[ConnectorSummary, ...] = ()


class ConnectorSignedIn(Payload):
    """A browser session captured for a source, proved against it, and stored.

    Carries no part of the credential and never will. The session itself is the sync account's
    whole identity at that company, so what is reported is what the instance said about it —
    who it belongs to and when it was taken — and where it now lives.
    """

    name: str = Field(description="The configured source the session was captured for.")
    base_url: str
    account: str = Field(description="Who the instance said the session belongs to.")
    captured_at: str
    expires_at: str = Field(
        description="When manicule will stop using it without a fresh sign-in. Its own ceiling "
        "(``session_max_age_hours``) rather than the instance's, which no cookie states."
    )
    stored_in: str = Field(description="Where the session was put.")
    forgotten: bool = Field(
        default=False, description="Whether this removed a stored session instead of taking one."
    )


# --- configuration -------------------------------------------------------------------------


class ConfigValue(Payload):
    """One configuration key, or the whole tree when no key was named.

    Always the **redacted** view. Returning the live object would hand out every API key,
    OAuth client secret and webhook signing key to anyone allowed to read configuration.
    """

    key: str = ""
    value: JsonValue = None
    source: str = Field(default="", description="Where the effective value came from.")


class ConfigChange(Payload):
    """One configuration key, changed."""

    key: str
    previous: JsonValue = None
    value: JsonValue = None
    path: str = Field(description="The file that was written.")


# --- workspaces ----------------------------------------------------------------------------


class WorkspaceSummary(Payload):
    """One workspace."""

    id: str
    name: str = ""
    mode: str = ""
    active: bool = False
    documents: int | None = Field(default=None, ge=0)


class WorkspaceList(Payload):
    """Every workspace this installation knows about."""

    active: str
    count: int = Field(ge=0)
    workspaces: tuple[WorkspaceSummary, ...] = ()


class WorkspaceSwitched(Payload):
    """The outcome of a switch."""

    previous: str
    active: str
    path: str = Field(description="The file that recorded it.")


# --- plugins -------------------------------------------------------------------------------


class ComponentSummary(Payload):
    """One registered component."""

    kind: str
    name: str
    plugin: str
    summary: str = ""


class PluginSummary(Payload):
    """One installed plugin."""

    name: str
    version: str = ""
    core_version: str = ""
    summary: str = ""
    enabled: bool = True
    components: tuple[ComponentSummary, ...] = ()


class AvailablePlugin(Payload):
    """One plugin offered by the community registry."""

    name: str
    version: str = ""
    core_version: str = ""
    summary: str = ""
    package: str = ""
    url: str = ""
    installed: bool = False
    compatible: bool = True
    incompatible_reason: str = ""


class PluginList(Payload):
    """What is installed, and what the registry offers."""

    count: int = Field(ge=0)
    plugins: tuple[PluginSummary, ...] = ()
    disabled: tuple[str, ...] = ()
    available: tuple[AvailablePlugin, ...] = ()
    registry_url: str = ""
    registry_error: str = ""


class PluginChanged(Payload):
    """The outcome of enabling or disabling a plugin."""

    name: str
    enabled: bool
    installed: bool
    path: str = ""
    detail: str = ""


# --- operations the command line owns ------------------------------------------------------


class BackupReport(Payload):
    """Where a backup went, and what it contains."""

    path: str
    created_at: str = ""
    files: int = Field(default=0, ge=0)
    bytes: int = Field(default=0, ge=0)
    schema_revision: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class RestoreReport(Payload):
    """What a restore put back."""

    path: str
    data_dir: str
    files: int = Field(default=0, ge=0)


class ExportReport(Payload):
    """A corpus export."""

    path: str
    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)


class ImportReport(Payload):
    """A corpus import."""

    path: str
    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)


class ResetReport(Payload):
    """What ``reset-index`` removed."""

    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    vectors_removed: bool = False


class InitReport(Payload):
    """What ``init`` decided and wrote."""

    path: str
    data_dir: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    hardware: dict[str, JsonValue] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    weights_pending: bool = Field(
        default=False,
        description="Whether the embedding weights are still to be downloaded, so the first "
        "`index` spends minutes fetching before it indexes anything. A field rather than a "
        "sentence a renderer finds in `notes`: it is the one fact that changes what the next "
        "command feels like, and every surface should be able to say so in its own voice. "
        "`False` also covers 'could not be determined' — a backend manicule knows no artefact "
        "route for is not one it may announce a download on behalf of.",
    )


class ServerAddress(Payload):
    """Where a server is listening, and whether that is loopback."""

    transport: str
    host: str = ""
    port: int | None = None
    loopback: bool = True
    tools: int = Field(default=0, ge=0)


class UpgradeReport(Payload):
    """What an upgrade would do, or did."""

    current: str
    target: str = ""
    latest: str | None = None
    backup: str | None = None
    performed: bool = False
    detail: str = ""


class CompletionScript(Payload):
    """A shell completion script."""

    shell: str
    script: str


class ApiKeySummary(Payload):
    """One API key. The secret itself appears exactly once, at creation."""

    id: str
    name: str
    prefix: str
    role: str
    workspace: str
    created_at: str = ""
    expires_at: str | None = None
    revoked: bool = False


class ApiKeyIssued(Payload):
    """A freshly minted key, with the only copy of its secret."""

    key: ApiKeySummary
    secret: str = Field(
        description="Shown once and never stored. Only its SHA-256 digest is kept, so a lost "
        "key is reissued rather than recovered."
    )


class ApiKeyList(Payload):
    """Every key in this workspace."""

    count: int = Field(ge=0)
    keys: tuple[ApiKeySummary, ...] = ()


class ApiKeyRevoked(Payload):
    """The outcome of a revocation."""

    id: str
    name: str
    revoked: bool


__all__ = [
    "CONTRACT_VERSION",
    "Anchored",
    "AnswerCitation",
    "AnswerResultPayload",
    "ApiKeyIssued",
    "ApiKeyList",
    "ApiKeyRevoked",
    "ApiKeySummary",
    "AuditEntry",
    "AuditPage",
    "AuthProviders",
    "AvailablePlugin",
    "BackupReport",
    "Check",
    "CheckState",
    "CollectionDeleted",
    "CollectionList",
    "CollectionMembership",
    "CollectionSummary",
    "CompletionScript",
    "ComponentSummary",
    "ConfigChange",
    "ConfigValue",
    "ConnectorList",
    "ConnectorSummary",
    "ConversationDeleted",
    "ConversationList",
    "ConversationMessages",
    "ConversationRenamed",
    "ConversationSummary",
    "ConversationTurn",
    "Diagnosis",
    "DocumentChunk",
    "DocumentDeleted",
    "DocumentDetail",
    "DocumentList",
    "DocumentReindexed",
    "DocumentRestored",
    "DocumentSummary",
    "DocumentTags",
    "Envelope",
    "ErrorInfo",
    "ExportReport",
    "FeedbackRecorded",
    "Identity",
    "ImportReport",
    "IndexStatus",
    "IngestReport",
    "InitReport",
    "Payload",
    "PluginChanged",
    "PluginHealth",
    "PluginHealthReport",
    "PluginList",
    "PluginSummary",
    "QueryLogEntry",
    "QueryLogPage",
    "ResetReport",
    "RestoreReport",
    "SearchHit",
    "SearchQuality",
    "SearchResult",
    "ServerAddress",
    "ShareCreated",
    "ShareRevoked",
    "SharedCitationLabel",
    "SharedConversation",
    "SharedTurnPayload",
    "Stats",
    "TagDeleted",
    "TagList",
    "TagSummary",
    "TrashList",
    "TrashedDocument",
    "UpgradeReport",
    "Workbench",
    "WorkbenchBlock",
    "WorkspaceList",
    "WorkspaceSummary",
    "WorkspaceSwitched",
    "failed",
    "succeeded",
]
