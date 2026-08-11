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


type CheckState = Literal["ok", "degraded", "failing", "unknown"]
"""How a diagnostic came out.

``unknown`` is not a fourth severity — it is the honest answer for a check that could
not run, and is deliberately distinct from ``ok``, which claims something was measured.
"""


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
    "AvailablePlugin",
    "BackupReport",
    "Check",
    "CheckState",
    "CompletionScript",
    "ComponentSummary",
    "ConfigChange",
    "ConfigValue",
    "ConnectorList",
    "ConnectorSummary",
    "Diagnosis",
    "DocumentChunk",
    "DocumentDeleted",
    "DocumentDetail",
    "DocumentList",
    "DocumentReindexed",
    "DocumentSummary",
    "Envelope",
    "ErrorInfo",
    "ExportReport",
    "ImportReport",
    "IndexStatus",
    "IngestReport",
    "InitReport",
    "Payload",
    "PluginChanged",
    "PluginList",
    "PluginSummary",
    "ResetReport",
    "RestoreReport",
    "SearchHit",
    "SearchResult",
    "ServerAddress",
    "Stats",
    "UpgradeReport",
    "WorkspaceList",
    "WorkspaceSummary",
    "WorkspaceSwitched",
    "failed",
    "succeeded",
]
