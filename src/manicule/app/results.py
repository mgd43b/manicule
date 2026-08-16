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
    Whether the operation completed successfully. Most failures carry only ``error``; an
    incomplete ingest also retains its partial counters in ``data`` so retry automation does
    not have to choose between the failure signal and the work already committed.

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

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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


type LifecyclePhase = Literal[
    "acquiring",
    "verifying",
    "rebuilding",
    "reembedding",
    "resetting",
    "deleting",
    "complete",
    "failed",
    "canceled",
]
type LifecycleOutcome = Literal[
    "running", "complete", "bounded", "deferred", "incomplete", "refused", "failed", "canceled"
]
type LifecycleRefusalCode = Literal[
    "capacity",
    "snapshot_not_promoted",
    "snapshot_changed",
    "workspace_scope_changed",
    "missing_local_input",
    "memory_bound",
    "temp_disk_bound",
    "invalid_replacement",
    "derivation_failed",
    "confirmation_required",
    "retention_policy",
    "shared_reference",
]
type LifecycleResource = Literal[
    "",
    "journal_records",
    "journal_metadata_bytes",
    "acquired_blob_backlog_bytes",
    "disk_headroom_bytes",
]


class LifecycleRefusal(Payload):
    """Typed, aggregate-only reason lifecycle work cannot proceed.

    Values are deliberately counts and resource names. Source identifiers, paths, URIs,
    exception messages and content have no place in a status object that is returned by every
    automation surface and may be retained by a scheduler.
    """

    code: LifecycleRefusalCode
    count: int = Field(default=0, ge=0)
    resource: LifecycleResource = ""
    limit: int | None = Field(default=None, ge=0)
    used: int | None = Field(default=None, ge=0)
    requested: int | None = Field(default=None, ge=0)


class LifecycleProgress(Payload):
    """One aggregate status vocabulary shared by snapshot and derived operations.

    Every field is safe to serialize, log and persist. Identities are opaque durable ids or
    fingerprints, never source ids. Optional values mean the operation cannot know the fact;
    they are not silently replaced with a plausible-looking zero or ``False``.
    """

    phase: LifecyclePhase
    outcome: LifecycleOutcome
    dry_run: bool = False

    enumerated_items: int = Field(default=0, ge=0)
    acquired_items: int = Field(default=0, ge=0)
    reused_items: int = Field(default=0, ge=0)
    omitted_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    pending_items: int = Field(default=0, ge=0)

    snapshot_completeness: Literal["", "complete", "partial"] = ""
    reproducibility_policy: str = ""
    snapshot_identity: str = ""
    snapshot_promoted: bool | None = None
    source_generation_identity: str = ""
    derived_generation_identity: str = ""
    candidate_watermark_present: bool | None = None
    committed_watermark_present: bool | None = None

    backlog_items: int = Field(default=0, ge=0)
    backlog_bytes: int = Field(default=0, ge=0)
    oldest_backlog_age_seconds: float | None = Field(default=None, ge=0)
    can_continue_offline: bool = False

    rate_items_per_second: float = Field(default=0, ge=0)
    estimated_remaining_items: int = Field(default=0, ge=0)
    estimated_remaining_seconds: float = Field(default=0, ge=0)
    refusal: LifecycleRefusal | None = None

    @model_validator(mode="after")
    def terminal_and_refusal_are_consistent(self) -> Self:
        if self.outcome == "refused" and self.refusal is None:
            raise ValueError("a refused lifecycle outcome requires a typed refusal")
        if self.outcome != "refused" and self.refusal is not None:
            raise ValueError("a lifecycle refusal requires the refused outcome")
        if self.outcome in {"complete", "failed", "canceled"} and self.pending_items:
            raise ValueError("a terminal lifecycle outcome cannot retain pending items")
        return self


class ErrorInfo(Payload):
    """What went wrong, in the shape a caller can act on."""

    type: str = Field(description="The exception class name, e.g. ``PolicyError``.")
    message: str
    hint: str = Field(
        default="",
        description="What to do about it, when there is something specific to say.",
    )


class Envelope(BaseModel):
    """One operation's outcome, as it is serialized by both surfaces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = CONTRACT_VERSION
    op: str = Field(min_length=1)
    ok: bool
    workspace: str = Field(min_length=1)
    data: dict[str, JsonValue] | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def one_outcome_shape(self) -> Self:
        """Hold the data/error invariant at every constructor, not only in the helpers."""
        if self.ok:
            if self.data is None or self.error is not None:
                msg = "a successful envelope requires data and forbids error"
                raise ValueError(msg)
            return self
        if self.error is None:
            msg = "a failed envelope requires error"
            raise ValueError(msg)
        if self.data is None:
            return self
        partial = IngestReport.model_validate(self.data)
        if (
            self.op not in {"index_path", "index_changes", "connector_sync", "import"}
            or partial.outcome != "incomplete"
            or not partial.retry_required
            or partial.incomplete_reason != self.error
        ):
            msg = (
                "a failed envelope may retain data only for a retry-required incomplete "
                "IngestReport whose reason matches error"
            )
            raise ValueError(msg)
        return self

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


def failed(
    op: str, workspace: str, error: ErrorInfo, *, payload: IngestReport | None = None
) -> Envelope:
    """Wrap a failure, retaining a partial result when the operation produced one."""
    return Envelope(
        op=op,
        ok=False,
        workspace=workspace,
        data=payload.model_dump(mode="json") if payload is not None else None,
        error=error,
    )


type CheckState = Literal["ok", "degraded", "failing", "unknown"]
"""How a diagnostic came out.

``unknown`` is not a fourth severity — it is the honest answer for a check that could
not run, and is deliberately distinct from ``ok``, which claims something was measured.

Declared here, above every payload that uses it, rather than beside :class:`Check`. Pydantic
resolves an annotation against the module namespace as the model class is built, so a payload
defined earlier in the file than its own type alias fails at import — loudly, but at import,
which is a poor place to discover an ordering rule.

These four names are the **only** status vocabulary manicule has. ``doctor --json`` reports
exactly the words the terminal prints, because a machine-readable contract whose statuses are
spelled differently from the human output is a trap for whoever automates against it: they read
the screen, write ``error``, and match nothing forever. ``docs/surfaces.md`` §5 records the
translation to the conventional ``ok``/``warning``/``error`` triple for consumers that need one.
"""

DOCTOR_SCHEMA_VERSION = 1
"""The shape of :class:`Diagnosis`, versioned separately from manicule itself.

Distinct from ``Envelope.version`` on purpose. That one moves with every release whether or not
any shape changed; this one moves only when the diagnosis payload gains, loses or repurposes a
field. A consumer pinning behavior wants the second, and giving it only the first forces it to
diff release notes.
"""


def redacted_path(path: str | Path) -> str:
    """A filesystem path with the account's home directory replaced by ``~``.

    ``doctor``'s output is the thing an operator pastes into an issue, a support thread or a
    chat window, and the paths in it run through ``$HOME`` — cache directories, the data
    directory, the configuration file. The home directory's *name* is the account name, which
    is a credential's worth of a hint on a shared or corporate machine and is never the part
    anybody needed.

    ``~`` rather than a fixed token, because the requirement is redaction that stays
    *diagnosable*: an operator reading their own output can still ``cd`` to what it names, and
    ``chmod 0700 ~/…`` is a command they can paste back. A path outside the home directory is
    returned unchanged — ``/srv/manicule`` names no account and hiding it would cost the reader
    the only location in the message.
    """
    text = str(path)
    try:
        home = str(Path.home())
    except (OSError, RuntimeError):  # pragma: no cover - a platform with no home to resolve
        return text
    # A home of "/" would turn every absolute path into a tilde, which is redaction that has
    # eaten the message rather than the account name.
    if not home or home == os.sep:
        return text
    if text == home:
        return "~"
    if text.startswith(home + os.sep):
        return "~" + text[len(home) :]
    return text


# --- retrieval and answers -----------------------------------------------------------------


class SourceReference(Payload):
    """Where a cited document was published, and where this installation keeps its copy.

    Present only for a document that carries authoritative source metadata — a locally mirrored
    page with a sidecar manifest, or any connector that supplies the same record. ``None``
    everywhere else, which is the honest answer for an ordinary file: there is no canonical
    address to report, and an empty string would read as one that had been looked for and found
    blank.

    **Both identities are here, and neither is presented as the other.** ``title``,
    ``canonical_uri``, ``source_id``, ``version`` and ``modified_at`` describe the *publication*.
    ``snapshot_path``, ``snapshot_checksum`` and ``retrieved_at`` describe *this machine's copy*.
    ``indexed_at`` is neither — it is when manicule indexed the copy. A consumer that wants to
    show a reader where to go uses the first group; an audit that wants to know what was actually
    read uses the second; and reproducing a result months later needs both, which is why the
    citation carries both rather than choosing.

    The flat shape is deliberate over nesting the two groups. These fields are rendered into
    one-line citations by three surfaces and a template, and a consumer reaching
    ``provenance.snapshot.path`` through two optional levels has two places to get a null check
    wrong.

    Carried as ``provenance`` on every payload that has one — **not** as ``source``, which is
    already taken on :class:`DocumentSummary` and means the name of the connector that owns the
    document. Two different senses of one word on one model is a field somebody reads wrong once
    and then relies on.
    """

    title: str = Field(default="", description="The document's own title, as its source has it.")
    canonical_uri: str = Field(
        default="", description="Where a reader goes to see the published document."
    )
    source_id: str = Field(
        default="", description="The identifier the publisher assigns and does not recycle."
    )
    version: str = Field(default="", description="The source's own version, compared as a string.")
    content_type: str = Field(
        default="",
        description="The media type the **source** published, which is not always the media "
        "type this installation stored. A page served as one thing and mirrored to a file "
        "whose suffix says another has two answers, and the document's own ``media_type`` is "
        "the local one; this is the publisher's.",
    )
    modified_at: str | None = Field(
        default=None,
        description="When the document was last edited **at its source**. Never this "
        "installation's ingestion time; see ``indexed_at``, which is that and is separate.",
    )
    section_path: tuple[str, ...] = Field(
        default=(),
        description="Where the document sits in its source's hierarchy, coarsest first. The "
        "passage's own position within the document is ``heading_path`` on the citation, and the "
        "two are not concatenated here — a consumer that wants a full section path joins them, "
        "and one that wants to say which manual a page came from does not have to unpick it.",
    )
    snapshot_path: str = Field(
        default="",
        description="Where the local copy sits, relative to the ingestion root. Relative so a "
        "citation reproduces elsewhere and does not publish this machine's directory layout.",
    )
    snapshot_checksum: str = Field(
        default="",
        description="Digest of the local copy's bytes — the same value as the document's "
        "``content_hash``, reported here so an audit reading a citation has it in hand.",
    )
    retrieved_at: str | None = Field(
        default=None, description="When the local copy was taken, as whoever took it declared."
    )
    indexed_at: str | None = Field(
        default=None, description="When this installation last indexed the copy."
    )
    unavailable_reason: str = Field(
        default="",
        description="Why there is no authoritative record, when one was attempted and refused — "
        "a malformed manifest, an unusable canonical URI. Reported rather than swallowed: the "
        "symptom of a silently ignored manifest is a citation that names a file, which is "
        "indistinguishable from having written no manifest at all.",
    )


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
    provenance: SourceReference | None = Field(
        default=None,
        description="The document's authoritative source metadata, when it has any. ``title`` "
        "and ``uri`` above are already the canonical ones where a record exists — this is the "
        "structured form, for a consumer that needs the version it cited or the snapshot it "
        "was read from rather than a line to display.",
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


class GlossaryExpansion(Payload):
    """A glossary term the query named, expanded, with where the definition came from.

    **The provenance fields have no defaults**, and that is the requirement rather than a
    style choice. ``bugs/bug2.md`` §3 forbids presenting an expansion without citation
    provenance, so the payload is built so there is no shape of it that omits the source: a
    surface cannot forget to include the document, because pydantic will not construct one
    without it.

    :attr:`provenance` is the exception to that and is nullable for the reason
    :class:`SourceReference` gives: most documents have no authoritative record, and ``null``
    says so where an empty object would read as one that was looked for and found blank. It is
    a *second* identity for the same document, never a substitute for the four above.

    Not an :class:`Anchored`. An anchor describes a *quotation*, and this is not one — nothing
    here is a passage of text, so an anchor on it would be a location for a span that is never
    shown. :attr:`chunk_id` is the citation, and it resolves through exactly the machinery every
    other citation resolves through.
    """

    document_id: str
    chunk_id: str
    uri: str
    title: str
    provenance: SourceReference | None = Field(
        default=None,
        description="The defining document's authoritative source metadata, when it has any, "
        "in the same shape a search hit and an answer citation report it. Here rather than left "
        "to be joined from a hit, because an expansion is reportable on results that contain no "
        "hit for it at all — a term whose definition was found and then kept out of the context "
        "is exactly that case — and the source identity of the document a definition was read "
        "from should not depend on whether the passage happened to survive.",
    )

    acronym: str
    """The normalized key that fired — ``NOW``."""

    display: str
    """The term as the source document writes it. Shown in preference to the normalized key,
    so a reader sees the document's own words rather than ours."""

    expansion: str
    matched: str = Field(
        default="", description="The token as the query wrote it, which may differ in case."
    )
    reason: str = Field(
        default="",
        description="Which rule admitted the occurrence: an exact-case match, a definitional "
        "question, or a term that is not an ordinary English word. Surfaced because these are "
        "the rules that stop the feature rewriting every use of a common word, and a rule "
        "nobody can see fire is a rule nobody can check.",
    )
    form: str = Field(default="", description="The written form the definition was read in.")
    detection_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly the source text reads as a definition. Never a claim about "
        "whether the definition is correct.",
    )
    location: str = Field(default="", description="Where in the document, in its own terms.")


class GlossaryConflict(Payload):
    """One term with more than one definition in scope, and every candidate.

    Reported instead of an expansion, never alongside one. A surface that showed a conflict and
    an expansion for the same term would be presenting a choice it had already made.
    """

    acronym: str
    matched: str = ""
    candidates: tuple[GlossaryExpansion, ...] = Field(
        default=(),
        min_length=2,
        description="Every definition in scope, each with its own provenance. At least two, "
        "because one is not a conflict — and the whole value of this field is that a reader "
        "can go and look at both documents.",
    )


class Glossed(Payload):
    """A result the glossary was consulted for, and what it had to say about the query.

    Carried by ``search`` and by ``ask``, and defined once rather than per payload because the
    three fields are one contract: :attr:`explicit_definition` is a claim *about*
    :attr:`expansions`, and a copy of the rule below in each payload is a rule that holds in
    one of them after somebody edits the other.

    **This is a stable public contract.** ``docs/surfaces.md`` §5 documents it as such: the
    boolean is flat and defaults to ``false``, so a client written before it existed parses a
    payload carrying it, and a client written after it parses one that does not.
    """

    expansions: tuple[GlossaryExpansion, ...] = Field(
        default=(),
        description="Glossary terms the query named, each with the document and passage its "
        "definition was read out of. A result that was retrieved through an expansion has to "
        "be able to say so: the reader is entitled to know that the search ran on words they "
        "did not type, and which document put them there.",
    )
    conflicts: tuple[GlossaryConflict, ...] = ()
    explicit_definition: bool = Field(
        default=False,
        description="Whether this result answers a question about a term by showing that "
        "term's definition. **A classification, not a quantity**: it is copied from "
        "``Confidence.explicit_definition`` and enters no arithmetic, so ``confidence``, its "
        "band and its components are byte-for-byte what they were before this field existed. "
        "It is machine-readable on purpose — ``confidence_reason`` says the same thing in "
        "English, and a client that parsed prose to find it would be reading a sentence "
        "written for a person. ``false`` for an ordinary use of a defined token, for a "
        "contested term, when the defining passage did not survive into the delivered "
        "context, and whenever the corpus was not consulted at all.",
    )

    @model_validator(mode="after")
    def _a_cited_definition_is_one_the_reader_can_open(self) -> Self:
        """Refuse the one combination that would be a claim without a citation.

        Requirement 9 of the specification says an explicit definition must resolve to a
        document, a chunk, a title and a URI. Every one of those lives on
        :class:`GlossaryExpansion`, which cannot be built without them — so the whole of the
        requirement reduces to *there is at least one expansion here*, and that is checkable by
        the model rather than by whichever test remembered to look.

        It has teeth because the empty case is reachable. A glossary entry whose document this
        workspace can no longer see is **dropped** from :attr:`expansions` rather than shown
        with a blank source, and retrieval classified the query before that drop happened. So
        the state this refuses is one a race really can produce, and the service resolves it by
        withdrawing the claim rather than by shipping an uncitable one.

        What it does not check, said plainly: *which* expansion was the defining one. With two
        terms named, one of them definitional, this passes on the presence of either. The
        narrower fact is not on the payload, and inventing a second opinion about it here —
        from the query text, or by re-reading the context — would be exactly the derived
        classification requirement 3 forbids.
        """
        if self.explicit_definition and not self.expansions:
            msg = (
                "explicit_definition is true and no glossary expansion is reported, so the "
                "result claims a definition was cited and names no document, chunk, title or "
                "URI for it. A claim a reader cannot go and check is worse than no claim: "
                "report false instead."
            )
            raise ValueError(msg)
        return self


class SearchResult(Glossed):
    """What ``search`` produced."""

    query: str
    profile: str
    count: int = Field(ge=0)
    hits: tuple[SearchHit, ...] = ()
    confidence: float | None = None
    confidence_band: str | None = None
    confidence_reason: str = ""
    expanded_query: str = Field(
        default="",
        description="The second query form glossary lookup produced, or empty when none did. "
        "The original is :attr:`query` and is never replaced.",
    )
    route: str = ""
    cached: bool = False
    truncated: bool = False
    elapsed_ms: int = Field(default=0, ge=0)

    collections: tuple[str, ...] = ()
    """The collections this search was restricted to, named as the caller named them.

    Empty means the search was not restricted to any, which is a different claim from a
    restriction that matched nothing — that one comes back as zero hits with the scope still
    reported here. Without it a scoped search and a workspace-wide search that happened to
    return the same passages are indistinguishable in a log.
    """


class AnswerCitation(Anchored):
    """A citation the answer carries, after verification."""

    slot: int = Field(ge=1)
    quote: str = ""
    verification: str = ""


class AnswerResultPayload(Glossed):
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
    provenance: SourceReference | None = Field(
        default=None,
        description="The document's authoritative source metadata, when it has any. On the "
        "summary as well as on a citation, because ``document list`` is where an operator looks "
        "to find out whether a manifest was honored — and a refused one says so here.",
    )


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
    """``reindexed``, ``superseded``, ``failed`` or ``unchanged``. Four words, and no others.

    ``superseded`` is the one that is neither a success nor a problem: a connector sync
    committed newer bytes for this document while it was being re-parsed, so the commit was
    declined and the corpus holds the newer text. Distinct from ``unchanged``, which says the
    re-parse ran and produced what was already stored — the opposite claim, and the one a reader
    stops looking after.
    """

    chunks: int = Field(ge=0)
    detail: str = ""


class EmbeddingCost(Payload):
    """What an operation cost at the embedder, in facts that are not each other.

    Reported separately from the chunk counts beside it because the two are routinely
    confused, and the confusion is expensive in exactly one direction: a chunk keeping its
    content-derived id is not a vector surviving, and a vector surviving is not a forward pass
    avoided. What an accelerator spends its time on is ``forward_calls``, and nothing above it
    in a sweep report is a measurement of that.
    """

    reused: int = Field(default=0, ge=0)
    """Vectors taken from the store without a model call, because the embedding input matched."""

    embedded: int = Field(default=0, ge=0)
    """Chunks handed to the embedder."""

    input_changed: int = Field(default=0, ge=0)
    """Of ``embedded``, those the index held an embedding input for and no longer matches.

    Includes a chunk whose ``text`` did not move — and which therefore kept its id and its
    citations — while the heading breadcrumb in its ``embed_text`` did. Kept apart from
    ``first_seen`` because a corpus that changed and a corpus that grew are different findings.
    """

    first_seen: int = Field(default=0, ge=0)
    """Of ``embedded``, chunks the index has no previous embedding input for at all.

    Growth rather than change. Nothing was reused for them because there was never anything to
    reuse.
    """

    repaired: int = Field(default=0, ge=0)
    """Of ``embedded``, those whose input was unchanged and whose stored vector was unusable.

    Missing, of the wrong dimension, or described by metadata that contradicts the chunk beside
    it. Found by reading the row rather than by trusting what it claims.
    """

    forward_calls: int = Field(default=0, ge=0)
    """Batches the embedder was asked for. The number an accelerator's time is proportional to."""

    cache_hits: int = Field(default=0, ge=0)
    """Chunks the embedder served from its in-memory cache rather than the model.

    The warm layer above durable reuse, reported apart from it because the two are routinely
    read as one. ``reused`` survives a process restart; this does not.

    There is deliberately no counter for reuse missed because the embedding *fingerprint*
    changed. That does not produce misses — it refuses the run and names the price — so such a
    counter could only ever read zero.
    """

    vectors_new: int = Field(default=0, ge=0)
    """Of ``embedded``, those the store held no row for.

    Not every row the operation wrote: a reused vector re-filed under a new chunk id also lands
    in a row that did not exist, and is counted under ``reused``. This pair says where the
    *embedder's* output went.
    """

    vectors_replaced: int = Field(default=0, ge=0)
    """Of ``embedded``, those that overwrote a row the store already had."""

    vectors_backfilled: int = Field(default=0, ge=0)
    """Reused rows that carried no recorded embedding-input identity and had one reconstructed.

    The one-time migration of a ``vectors/`` directory that predates the identity column,
    counted as it happens. It costs no forward pass, and it reaches zero once every row a sweep
    touches has been written since.
    """


class ReembedPlanReport(Payload):
    """Aggregate-only offline vector migration price; no source or storage identifiers."""

    dry_run: bool = True
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    input_bytes: int = Field(ge=0)
    estimated_seconds: float = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    temporary_disk_bytes: int = Field(ge=0)
    unrepairable_documents: int = Field(ge=0)
    target_identity: str = Field(description="SHA-256 identity of the exact target embedder.")
    target_dimension: int = Field(gt=0)
    lifecycle: LifecycleProgress


class ReembedRunReport(Payload):
    """Aggregate durable progress for one operator-created re-embedding run."""

    run_id: str
    state: str
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    documents_completed: int = Field(ge=0)
    chunks_completed: int = Field(ge=0)
    estimated_remaining_seconds: float = Field(ge=0)
    retry_required: bool
    terminal: bool
    published: bool
    target_identity: str
    target_dimension: int = Field(gt=0)
    lifecycle: LifecycleProgress


class ReembedCleanupReport(Payload):
    """Whether terminal shadow storage was physically removed."""

    run_id: str
    removed: bool


class RebuildPlanReport(Payload):
    """Aggregate-only dry run for one promoted retained snapshot."""

    generation_id: str
    snapshot_id: str
    documents: int = Field(ge=0)
    known_source_bytes: int = Field(ge=0)
    estimated_chunks: int = Field(ge=0)
    estimated_seconds: float = Field(ge=0)
    estimated_peak_memory_bytes: int = Field(ge=0)
    estimated_temporary_bytes: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    refusal_code: str | None = None
    runnable: bool
    lifecycle: LifecycleProgress


class RebuildRunReport(Payload):
    """Durable aggregate checkpoint for an offline replacement generation."""

    generation_id: str
    state: str
    expected_items: int = Field(ge=0)
    next_sequence: int = Field(ge=0)
    documents_built: int = Field(ge=0)
    chunks_built: int = Field(ge=0)
    vectors_reused: int = Field(ge=0)
    vectors_embedded: int = Field(ge=0)
    diagnostic_code: str | None = None
    lifecycle: LifecycleProgress


class LifecycleReport(Payload):
    """Aggregate-only plan or outcome for one explicit retention boundary."""

    operation: str
    dry_run: bool
    eligible_items: int = Field(default=0, ge=0)
    eligible_bytes: int = Field(default=0, ge=0)
    protected_items: int = Field(default=0, ge=0)
    protected_bytes: int = Field(default=0, ge=0)
    snapshot_items: int = Field(default=0, ge=0)
    unrecoverable_items: int = Field(default=0, ge=0)
    unrecoverable_bytes: int = Field(default=0, ge=0)
    removed_items: int = Field(default=0, ge=0)
    released_bytes: int = Field(default=0, ge=0)
    confirmation: str | None = None
    source_contacted: bool = False
    lifecycle: LifecycleProgress = Field(
        default_factory=lambda: LifecycleProgress(phase="complete", outcome="complete")
    )


class StaleReparseReport(Payload):
    """What a corpus-wide re-parse of stale documents did.

    Counts rather than a document list. A sweep over a personal corpus names hundreds of ids
    that nobody reads and nothing can act on, while the two lists that *are* here name the
    documents an operator has to do something about — which is the whole difference between a
    report and a log.

    **No field carries document text.** Every string here is an id, a URI or a reason, because
    this payload is printed to a terminal, written to whatever a shell pipeline points at, and
    returned by an operation whose entire subject is retained content.
    """

    dry_run: bool = False
    """Whether this was a plan. A dry run reports ``selected`` and nothing it did."""

    selected: int = Field(default=0, ge=0)
    """Documents whose recorded parse fingerprint is not one an installed parser produces."""

    reparsed: int = Field(default=0, ge=0)
    """Documents rebuilt from retained bytes."""

    unchanged: int = Field(default=0, ge=0)
    """Of ``reparsed``, those that came out with the chunk ids they went in with.

    The expected majority after a rules bump, and the number that says the bump was narrow. A
    ``parse_fp`` records the parser's version rather than the document's content, so every
    document that parser produced is stale — and nothing can tell which ones come out different
    without parsing them.
    """

    changed: int = Field(default=0, ge=0)
    """Of ``reparsed``, those whose chunk set moved."""

    chunks_new: int = Field(default=0, ge=0)
    """Chunks produced that were not already stored, and so new rows in the vector store.

    Not the sweep's embedding cost either, and the two are no longer even close: a chunk reaches
    the model only when its *embedding input* is new, changed or has no readable vector, which
    is a different set from the chunks that are new. ``embedding`` is the cost, and
    ``docs/ingest.md`` §10.1 prices both.
    """

    chunks_kept: int = Field(default=0, ge=0)
    """Chunks that survived with their id, and so with the vector row already stored.

    Chunk ids are content-derived, so a chunk the re-parse did not move keeps its id, keeps the
    row every citation to it resolves through, and keeps that citation working. **That is all
    this says.** It is a count of vector *rows* kept, not of vectors kept, and not of embedding
    work avoided — a chunk in here whose heading breadcrumb moved has the id it always had, the
    row it always had, and a vector inside it that can no longer be used. ``embedding`` is what
    was avoided, measured at the model.
    """

    embedding: EmbeddingCost = Field(default_factory=EmbeddingCost)
    """What the sweep cost at the embedder, summed over the documents it rebuilt."""

    unrepairable: int = Field(default=0, ge=0)
    """Documents that cannot be re-parsed, because there are no retained bytes to read."""

    failed: int = Field(default=0, ge=0)
    """Documents whose re-parse was attempted and did not finish."""

    unrepairable_documents: tuple[str, ...] = ()
    """One line per unrepairable document: which it is, why, and what would repair it."""

    failures: tuple[str, ...] = ()
    """One line per failure. Neither list fails the sweep — the rest of the corpus completes."""

    superseded: int = Field(default=0, ge=0)
    """Documents a newer revision overtook while this sweep was re-parsing them.

    Neither ``reparsed`` nor ``failed``, and the point of a third number is that it is neither.
    A connector sync committed newer bytes for the document mid-repair, and the guard at the
    commit refused to write the older parse back over them — so the corpus holds the newer text
    and this sweep left it alone. There is nothing for an operator to do about one.
    """

    superseded_documents: tuple[str, ...] = ()
    """One line per superseded document: which it is and what overtook it. No document text."""


class StaleGlossaryReport(Payload):
    """What a corpus-wide glossary recompute did.

    The counterpart to :class:`StaleReparseReport` at a rung below it, and the fields differ
    because the costs do. That report counts chunks, because a re-parse is charged in embedding
    passes. This one counts entries, because a recompute is charged in regular expressions over
    text already on disk — so the number an operator wants is not what it spent but what the
    corrected rules did to the vocabulary.

    **No field carries document text, and none carries a definition.** Every string here is a
    document id, a URI or an error, which matters more here than anywhere else on this module:
    the subject of this operation is the corpus's own vocabulary, and a report that named the
    terms it had removed would print the contents of the index to a terminal and to whatever a
    shell pipeline points at.
    """

    dry_run: bool = False
    """Whether this was a plan. A dry run reports ``selected`` and writes nothing at all."""

    selected: int = Field(default=0, ge=0)
    """Documents whose recorded glossary lineage is not what the installed detector produces.

    Includes every document with no recorded lineage, which on the first release with this
    column is the whole corpus. That is the migration policy rather than an accident: entries
    detected before anything recorded a detector cannot be shown to have come from the rules
    installed now, and trusting them indefinitely is what let five corrections to detection land
    against a corpus that reported itself current.
    """

    redetected: int = Field(default=0, ge=0)
    """Documents whose entries were recomputed from stored chunks. Zero on a dry run."""

    unchanged: int = Field(default=0, ge=0)
    """Of ``redetected``, those whose entry set came out exactly as it went in.

    The expected majority after a narrow rule change, and they still advance their fingerprint:
    the corpus now records which detector read them, and "the rules did not change this page" is
    a result rather than an absence of one.
    """

    changed: int = Field(default=0, ge=0)
    """Of ``redetected``, those whose entry set moved.

    Compared as sets rather than counts, because the change this exists for removes false
    entries and adds real ones on the same page.
    """

    entries_before: int = Field(default=0, ge=0)
    entries_after: int = Field(default=0, ge=0)
    """Entries across every document recomputed, before and after. Two totals, never a net."""

    failed: int = Field(default=0, ge=0)
    """Documents the detector could not read.

    Each one keeps the entries it had and keeps its stale lineage, so it is still selected next
    time and still reported by ``doctor``. Failing closed: a detector bug costs a repair, never
    a working glossary.
    """

    failures: tuple[str, ...] = ()
    """One line per failure, naming the document and the error. The sweep completes regardless."""

    unrepairable: int = Field(default=0, ge=0)
    """Documents whose chunks are gone, so there is nothing to recompute a glossary from.

    Reported rather than stamped. Detecting over no chunks yields no entries, which is a
    well-formed derived result — so recording it would convert a missing-chunks problem into an
    invisible empty-glossary one and take the document out of the selection for ever. These need
    a repair one rung up, and the line names it.
    """

    unrepairable_documents: tuple[str, ...] = ()
    """One line per unrepairable document: which it is, why, and what would fix it."""

    superseded: int = Field(default=0, ge=0)
    """Documents a newer sync overtook mid-recompute, so the write was declined.

    **Neither work done nor work to do**, and the same reading :class:`StaleReparseReport`
    applies one rung up: a sync committed newer chunks while this was reading older ones, so the
    corpus holds the newer state and this sweep did not touch it. Nothing needs doing — the
    document is selected again next time, against the chunks that overtook it.
    """

    superseded_documents: tuple[str, ...] = ()
    """One line per superseded document: which it is and what overtook it."""


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


class CollectionCounts(Payload):
    """How much is in a collection, counted now rather than remembered.

    Both numbers are live. A rule-driven collection has no stored membership to count, so a
    remembered total would be a number that was true on the day it was written and goes on
    being reported afterwards.
    """

    collection_id: str
    name: str
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)


class CollectionOrphans(Payload):
    """Live documents belonging to no collection, and what was done about them.

    ``deleted`` is false for the report, which is what a run produces unless deletion was
    asked for by name. Deletion is into the trash, so ``document_ids`` names documents that
    can still be restored rather than documents that are gone.
    """

    count: int = Field(ge=0)
    deleted: bool = False
    document_ids: tuple[str, ...] = ()


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

    available: bool = Field(description="Whether any judgments have been recorded at all.")
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


type IngestOutcome = Literal["complete", "bounded", "incomplete"]


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
    outcome: IngestOutcome = "complete"
    enumeration_completed: bool = True
    watermark_advanced: bool = False
    snapshot_completeness: Literal["", "complete", "partial"] = ""
    snapshot_omissions: int = Field(default=0, ge=0)
    snapshot_omission_reasons: dict[str, int] = Field(default_factory=dict)
    retry_required: bool = False
    derivation_deferred: bool = False
    intentionally_bounded: bool = False
    unrecorded: int = Field(default=0, ge=0)
    incomplete_reason: ErrorInfo | None = None
    elapsed_ms: int = Field(default=0, ge=0)
    lifecycle: LifecycleProgress = Field(
        default_factory=lambda: LifecycleProgress(phase="complete", outcome="complete")
    )


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
    weights_ref: str | None = Field(
        default=None,
        description="Exact hub commit or local digest whose bytes produced stored vectors.",
    )
    weights_identity: str | None = Field(
        default=None,
        description="Artifact compatibility identity; shared only by a qualified backend pair.",
    )
    chunk_fingerprint: str | None = None
    glossary: str = Field(
        default="",
        description="What the installed glossary detector is, in one line — or that detection "
        "is switched off. Here beside the other two because ``status`` is where somebody looks "
        "to find out what built this index, and until this line existed the answer named the "
        "three stages that had fingerprints and silently omitted the fourth.",
    )
    stale_glossary: int = Field(
        default=0,
        ge=0,
        description="Documents whose stored glossary lineage is not what the installed "
        "detector produces, ``NULL`` lineage included. A count over an indexed column, so it "
        "costs no glossary text to report. Non-zero is not a fault: it is what a detector fix "
        "looks like from the corpus's side, and `manicule document reindex --stale-glossary` "
        "is what clears it.",
    )
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
    """One diagnostic.

    ``name`` is the **stable identifier**: ``configuration``, ``transport``, ``plugins``,
    ``storage``, ``permissions``, ``index``, ``grammars``, ``vocabularies``, ``models``, and
    ``component:<kind>:<name>`` for anything already constructed. It is what a monitor selects
    on, so it is chosen once and does not move with the wording.

    ``detail`` and ``facts`` are the same finding twice, for two readers. ``detail`` is the
    sentence a person reads; ``facts`` is what a script would otherwise have to recover by
    parsing that sentence, which is how a wording change becomes somebody's outage.
    """

    name: str = Field(description="The stable identifier. Selected on; never reworded.")
    state: CheckState
    detail: str = Field(default="", description="The human-readable summary.")
    facts: dict[str, JsonValue] = Field(
        default_factory=dict[str, JsonValue],
        description="Structured form of what the check measured. Empty where the state and "
        "the summary are the whole finding.",
    )
    remedy: str = Field(
        default="",
        description="What to do about it — a command where there is one, otherwise the "
        "shortest actionable instruction. Empty on a healthy check, and on one whose repair "
        "depends on how the state was reached rather than on a step that can be named.",
    )


class Diagnosis(Payload):
    """Everything ``doctor`` looked at.

    ``state`` is the **worst** state among the checks, not a summary of them: a rollup that
    reported the best one would be a diagnosis nobody could act on.
    """

    state: CheckState = Field(description="The worst state among ``checks``.")
    schema_version: int = Field(
        default=DOCTOR_SCHEMA_VERSION,
        description="The shape of this payload. Moves only when the shape does.",
    )
    manicule_version: str = Field(
        default=CORE_VERSION, description="The manicule that produced the diagnosis."
    )
    checked_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="When the diagnosis was taken, ISO 8601 in UTC. A health record with no "
        "time on it cannot be told from a stale one somebody pasted.",
    )
    checks: tuple[Check, ...] = ()


# --- connectors ----------------------------------------------------------------------------


class ConnectorSummary(Payload):
    """One configured source."""

    name: str
    type: str
    enabled: bool = True
    installed: bool = True
    last_synced_at: str | None = None
    status: str = ""
    documents: int | None = Field(default=None, ge=0)
    last_outcome: IngestOutcome | None = None
    retry_required: bool = False
    last_error_type: str = ""
    last_enumeration_completed: bool | None = None
    last_watermark_advanced: bool | None = None
    last_lifecycle: LifecycleProgress | None = None


class ConnectorList(Payload):
    """Every configured source."""

    count: int = Field(ge=0)
    connectors: tuple[ConnectorSummary, ...] = ()


class SnapshotStatusReport(Payload):
    """Aggregate identity, integrity and progress for one durable source snapshot."""

    connector: str
    snapshot_id: str
    state: str
    verified: bool
    verification_performed: bool = False
    lifecycle: LifecycleProgress


class SidecarSkip(Payload):
    """One page that produced no manifest, and why."""

    path: str
    reason: str
    outcome: str = Field(
        default="",
        description="The machine-readable half of ``reason``: one of the adapter's outcome "
        "names. A surface counting refusals by kind should not have to match on prose, which "
        "is the half that gets rewritten.",
    )


class SidecarReport(Payload):
    """What one sidecar-generation run did.

    :attr:`skipped` carries every page that produced nothing, with its reason, rather than a
    count. A run over a directory whose files all lack a metadata section would otherwise report
    ``written: 0`` and look like a clean conversion of nothing — and the operator's next move
    depends entirely on whether the answer is "wrong directory", "wrong exporter" or "already
    done".
    """

    root: str
    source: str = Field(
        default="",
        description="The configured connector instance the root and profiles came from, or "
        "empty for a one-off conversion of a directory named directly.",
    )
    profiles: tuple[str, ...] = Field(
        default=(),
        description="The enriched-document profiles this run recognized, in the precedence "
        "order they were applied in.",
    )
    """Reported because it is the answer to the question a disappointing run raises.

    ``no_profile`` for every page means "none of these matched what I was looking for", and the
    operator's next move depends entirely on *what it was looking for* — the built-in default, or
    the two profiles their source declares. Leaving that to be inferred from whether ``--source``
    was typed is how a conversion run under the wrong profile set looks identical to a directory
    of ordinary HTML.
    """

    considered: int = Field(default=0, ge=0)
    written: int = Field(default=0, ge=0)
    skipped: tuple[SidecarSkip, ...] = ()
    by_outcome: dict[str, int] = Field(
        default_factory=dict[str, int],
        description="How many files reached each outcome, adapted ones included. Every "
        "considered file appears in exactly one bucket, so the counts sum to ``considered`` and "
        "a run that adapted nothing says which kind of nothing it was.",
    )
    """The counters, keyed by outcome, and they are not a summary of :attr:`skipped`.

    A caller that wanted "how many files matched no profile" would otherwise have to group the
    skip list by parsing its reasons. That is the failure ``detail``/``facts`` exists to prevent
    one layer up: the sentence is for a person and the bucket is for a program, and deriving the
    second from the first makes a wording change into somebody's broken dashboard.
    """


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
    """What ``reset-index`` rebuilt while retaining authoritative source state."""

    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    vectors_removed: bool = False
    snapshots_retained: int = Field(default=0, ge=0)


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
        "`False` also covers 'could not be determined' — a backend manicule knows no artifact "
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
    "DOCTOR_SCHEMA_VERSION",
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
    "GlossaryConflict",
    "GlossaryExpansion",
    "Glossed",
    "Identity",
    "ImportReport",
    "IndexStatus",
    "IngestOutcome",
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
    "ReembedCleanupReport",
    "ReembedPlanReport",
    "ReembedRunReport",
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
    "SourceReference",
    # `StaleReparseReport` is listed here for the first time, and it is not this change's.
    # #113 added the class and not the name, so `from manicule.app.results import *` produced a
    # module missing one of the two sweep reports — which nothing failed on, because the CLI
    # imports the module rather than its star. Adding one name beside it and leaving the gap
    # would make the omission look deliberate.
    "StaleGlossaryReport",
    "StaleReparseReport",
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
    "redacted_path",
    "succeeded",
]
