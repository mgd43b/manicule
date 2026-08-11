"""What an answer carries: citations, the drops, and the accounting.

Every type here exists to keep two numbers apart. :class:`~manicule.core.retrieval.Context`
arrives with a :class:`Confidence` about **retrieval support**, computed before generation;
this module adds an entirely separate account of **whether the citations verified**. They are
surfaced side by side and never blended, because a blend answers neither question and would
also move with the generation model — destroying the property that makes confidence
comparable at all.

Nothing here is behind a protocol, and that is deliberate (``docs/generation.md`` §2.2). A
plugin supplies a provider; it does not get to supply the part of the system the ticket is
about.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.config.providers import Egress
from manicule.core.anchors import Anchor
from manicule.core.content import BlockKind
from manicule.core.generation import FinishReason, Usage


class Verification(StrEnum):
    """How far up the ladder a citation got.

    Three levels, and the level reached travels with the citation rather than being averaged
    away: a citation verified only to :attr:`LOCATED` is a weaker claim than one verified to
    :attr:`RESOLVED`, and reporting the same word for two different amounts of checking is
    how a guarantee stops meaning anything.
    """

    BOUND = "bound"
    """Level 0: the slot is an integer in ``1..len(context.passages)``.

    Free, and it catches invention — a model naming slot 9 of 5. It is the level that makes
    citation hallucination structurally impossible rather than merely unlikely: the model
    contributes a small integer and nothing else.
    """

    LOCATED = "located"
    """Level 1: the passage's anchor is not :class:`~manicule.core.anchors.Unlocated`.

    Free, and it catches a citation pointing nowhere by the parser's own admission.
    """

    RESOLVED = "resolved"
    """Level 2: the anchor resolves over the retained source bytes to text containing what
    the chunk claims.

    A blob read and a parse, cached. It catches anchors that have drifted from the document,
    missing or corrupt source bytes, and anchors written by a parser version that is no
    longer running. Not redundant with the parser conformance suite: that runs against
    fixtures rather than against this corpus, the bytes can be gone, a stored conversation
    replays its citations after a re-ingest, and a restore can leave anchors from code that
    no longer runs.
    """

    @property
    def level(self) -> int:
        """0, 1 or 2."""
        return _LEVELS[self]

    def at_least(self, floor: Verification) -> bool:
        """Whether this is as strong as ``floor``.

        A named method rather than ``>=``. These are ``str`` values, so the comparison
        operators already work and — by pure accident of the alphabet — ``"bound" < "located"
        < "resolved"`` even orders them correctly today. A member added later would break
        that silently, and nothing would fail: citations would simply start surviving or
        being dropped for the wrong reason.
        """
        return self.level >= floor.level


_LEVELS: dict[Verification, int] = {
    Verification.BOUND: 0,
    Verification.LOCATED: 1,
    Verification.RESOLVED: 2,
}


class DropReason(StrEnum):
    """Why a citation did not survive.

    Reported in band and persisted, never logged and forgotten: a warning line beside a
    normal-looking answer is the shape of defect this project keeps naming, and this is that
    shape one layer up.
    """

    OUT_OF_RANGE = "out_of_range"
    """The slot named no passage. Level 0 — the model invented it."""

    UNLOCATED = "unlocated"
    """The passage's parser could not place it, and said so."""

    UNRESOLVABLE = "unresolvable"
    """The anchor did not resolve to the text the chunk claims, or the bytes were gone."""

    VERIFICATION_TIMEOUT = "verification_timeout"
    """Verification did not finish inside its budget.

    Deliberately distinct from :attr:`UNRESOLVABLE`: one means the disk is slow and the other
    means the citation is wrong, and they call for different remedies. A nonzero rate of this
    is an operational defect a diagnostic reports, not a property of the corpus.
    """

    MALFORMED_MARKER = "malformed_marker"
    """A closed marker whose payload was not a slot list.

    Counted rather than shown. The ``cite:`` prefix exists so that a malformed attempt is
    still recognisable as an attempt instead of being mistaken for prose.
    """


class Citation(BaseModel):
    """One verified citation. **Not one field of this comes from the model.**

    The model contributes a small integer. Everything else is built from the passage that
    integer selected, which is what deletes an entire class of failure rather than mitigating
    it: a model cannot invent a page number because it never writes one, cannot mangle a path
    because it never sees one it could copy wrong, and cannot cite a document that was not
    retrieved because there is no slot for one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: int = Field(ge=1, description="Ordinal within this answer. Not an identifier.")
    document_id: str = Field(min_length=1)
    uri: str
    title: str
    heading_path: tuple[str, ...] = ()
    anchor: Anchor
    chunk_id: str = Field(min_length=1)
    kind: BlockKind = Field(
        default=BlockKind.PROSE,
        description="What the cited block is. Carried because *whether a breadcrumb may be "
        "disclosed* depends on it: a spreadsheet chunk's heading path is its sheet name and a "
        "code chunk's is its symbol chain, while a document's is the section title an "
        "attestation is supposed to name. Inferring it from the anchor type gets both wrong — "
        "Markdown emits a LineAnchor for prose.",
    )
    kind: BlockKind = Field(
        default=BlockKind.PROSE,
        description="What the cited block is. Carried because *whether a breadcrumb may be "
        "disclosed* depends on it: a spreadsheet chunk's heading path is its sheet name and a "
        "code chunk's is its symbol chain, while a document's is the section title an "
        "attestation is supposed to name. Inferring it from the anchor type gets both wrong — "
        "Markdown emits a LineAnchor for prose.",
    )
    quote: str = Field(
        description="``Chunk.text``, whole and byte for byte. Never ``embed_text``, which "
        "carries the retrieval breadcrumb; never normalised, because showing a "
        "whitespace-flattened rendering of a quotation is a change to the quotation; never "
        "trimmed, because an anchor describes the whole chunk and a trimmed quote makes the "
        "anchor claim more than the text says."
    )
    verification: Verification

    @property
    def label(self) -> str:
        """Title and breadcrumb, with no passage text.

        What an anonymous viewer of a shared conversation is shown, and what a compact
        interface renders inline.
        """
        trail = " › ".join(self.heading_path)  # noqa: RUF001 - the breadcrumb separator, not a comparison
        return f"{self.title} — {trail}" if trail else self.title


class CitationDrop(BaseModel):
    """A citation that did not survive verification, and why.

    Carried on the answer rather than logged, because the reader is the person who needs to
    know that a sentence they are reading lost its support.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: int | None = Field(
        default=None,
        description="The slot the model named, or ``None`` when the marker did not name one "
        "— a malformed attempt has no slot to report and still has to be counted.",
    )
    reason: DropReason
    reached: Verification | None = Field(
        default=None,
        description="The strongest level this citation did reach before failing, or ``None`` "
        "when it reached none. A drop at level 2 is a different diagnosis from a drop at "
        "level 0, and averaging them loses the difference.",
    )
    detail: str = ""

    @model_validator(mode="after")
    def _slots_and_reasons_agree(self) -> Self:
        if self.reason is DropReason.MALFORMED_MARKER:
            if self.slot is not None:
                msg = f"a malformed marker names no slot; got slot={self.slot!r}"
                raise ValueError(msg)
        elif self.slot is None:
            msg = f"drop reason {self.reason.value!r} must name the slot it dropped"
            raise ValueError(msg)
        return self


class PolicyDrop(BaseModel):
    """A passage removed from the context before the prompt was built.

    Surfaced exactly like a dropped citation. Refusing the whole query instead would make the
    mere *existence* of one restricted document break unrelated questions that happened to
    retrieve it at rank 7; dropping the passage answers from what policy permits and says
    what it could not use.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    reason: str = Field(
        min_length=1,
        description="Which rule removed it, in words an operator can act on.",
    )


class CitationAccounting(BaseModel):
    """Counts and reasons. **No score.**

    Objective and needing no labels, which is what makes it the first answer-side quality
    signal this project has. It is not a probability and is never combined with
    :class:`Confidence`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slots_offered: int = Field(default=0, ge=0)
    markers_seen: int = Field(default=0, ge=0)
    verified: int = Field(default=0, ge=0)
    dropped: int = Field(default=0, ge=0)
    unterminated_markers: int = Field(
        default=0,
        ge=0,
        description="Attempts that never closed inside the marker bound and were released "
        "into the answer verbatim. Not a drop — nothing was deleted — but not prose either.",
    )
    verification_cache_hits: int = Field(default=0, ge=0)

    @property
    def offered_no_citations(self) -> bool:
        """Whether the model emitted no markers at all.

        Recorded, not judged: an answer with no markers may be the correct answer ("the
        sources do not cover this"), and no mechanism can distinguish that from a model that
        forgot. The mechanism that eventually does is a human pressing ``citation-wrong``.
        """
        return self.markers_seen == 0


class AnswerEnvelope(BaseModel):
    """Everything the caller receives besides the text itself.

    **An interface must not render a single blended percentage from these.** That is not a
    suggestion about visual design: :attr:`confidence` answers "how well-supported is this by
    the corpus", computed before generation and comparable only within one pipeline identity,
    and :attr:`citations` answers "did the citations verify". A blend answers neither and
    moves with the generation model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(default="", description="The answer as it will be stored and shown.")
    confidence: float | None = Field(
        default=None,
        description="Retrieval's own number, forwarded **verbatim** or absent. Generation "
        "never writes to it: a high confidence with an ungrounded answer is a meaningful and "
        "diagnosable combination — the evidence was there and the model did not use it — and "
        "it is only visible because the two numbers stayed separate.",
    )
    citations: tuple[Citation, ...] = ()
    dropped: tuple[CitationDrop, ...] = ()
    policy_dropped: tuple[PolicyDrop, ...] = ()
    accounting: CitationAccounting = Field(default_factory=CitationAccounting)
    verification_level: Verification = Field(
        default=Verification.RESOLVED,
        description="The strongest level available this run. Degrades to "
        "``located`` when source bytes are not retained, and the answer says so rather than "
        "reporting the same word for two different amounts of checking.",
    )
    ungrounded: bool = Field(
        default=False,
        description="Every citation was dropped while the context was non-empty. A single "
        "failed marker is a model slip; all of them failing is a model that ignored its "
        "context wholesale, or a corpus whose bytes are gone. A flag, not a refusal — the "
        "answer may still be useful and the reader decides with the fact in front of them.",
    )
    corpus_consulted: bool = Field(
        default=True,
        description="False for a directly-routed answer, which carries no citations and says "
        "so. Such an answer is **not** ``ungrounded``: that means 'the context was non-empty "
        "and nothing survived', and this context was empty by design.",
    )
    context_truncated: bool = False
    finish_reason: FinishReason | None = None
    error: str | None = None
    usage: Usage | None = None
    egress: Egress = Field(
        default=Egress.IN_PROCESS,
        description="Where this answer's prompt went, as classified from the resolved "
        "endpoint. Recorded on every answer because the two residual limits — a loopback "
        "proxy, and a hosted provider trusted to be what its hostname says — are not "
        "detectable from inside the process, so what manicule believed has to be auditable.",
    )
    redacted: bool = Field(
        default=False, description="Whether redaction was applied to what was sent."
    )


class EventKind(StrEnum):
    """What one element of the answer stream is."""

    DELTA = "delta"
    """A run of answer text, already past the binder."""

    CITATION = "citation"
    """A citation that verified, emitted at the point its marker survived."""

    DROP = "drop"
    """A citation that did not, emitted at the point its marker was deleted."""

    FINAL = "final"
    """The envelope. Always last on a stream that completed or failed in band."""


class AnswerEvent(BaseModel):
    """One element of the answer stream.

    A single stream rather than text on one channel and citations on another, because the
    *position* of a citation within the text is information: a consumer rendering deltas in
    order sees each citation exactly where its marker was.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EventKind
    text: str = ""
    citation: Citation | None = None
    drop: CitationDrop | None = None
    envelope: AnswerEnvelope | None = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> Self:
        expected = {
            EventKind.DELTA: self.text != "",
            EventKind.CITATION: self.citation is not None,
            EventKind.DROP: self.drop is not None,
            EventKind.FINAL: self.envelope is not None,
        }[self.kind]
        if not expected:
            msg = f"an event of kind {self.kind.value!r} must carry its own payload"
            raise ValueError(msg)
        return self

    @classmethod
    def delta(cls, text: str) -> AnswerEvent:
        return cls(kind=EventKind.DELTA, text=text)

    @classmethod
    def cited(cls, citation: Citation) -> AnswerEvent:
        return cls(kind=EventKind.CITATION, citation=citation)

    @classmethod
    def dropped(cls, drop: CitationDrop) -> AnswerEvent:
        return cls(kind=EventKind.DROP, drop=drop)

    @classmethod
    def final(cls, envelope: AnswerEnvelope) -> AnswerEvent:
        return cls(kind=EventKind.FINAL, envelope=envelope)


class GenerationTrace(BaseModel):
    """One record per answer, beside the retrieval trace.

    **It never contains document text, query text, or matched values.** Recording which
    detector fired and how often is diagnostic; recording what it matched turns the trace
    into the leak the detector existed to prevent.

    A return value rather than a row in ``query_logs``: that table is whole-query product
    telemetry, and per-call detail there is a schema change in service of a consumer that
    keeps its results elsewhere.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- model
    model: str = Field(default="", description="The resolved provider/model string as sent.")
    endpoint: str | None = None
    egress: Egress = Egress.IN_PROCESS
    context_window: int = Field(default=0, ge=0)
    num_ctx_sent: int | None = Field(
        default=None,
        description="The window manicule demanded of a served local runtime, or ``None`` "
        "where none was demanded. Derived from the profile arithmetic rather than "
        "configured, so it cannot disagree with the budget it exists to satisfy.",
    )

    # --- budget
    estimated_prompt_tokens: int = Field(default=0, ge=0)
    true_prompt_tokens: int | None = Field(
        default=None,
        description="``None`` means ``usage_unavailable`` — which is not the same as 'the "
        "estimate was correct'. Treating silence as agreement is how a calibration loop "
        "reports success forever without ever running.",
    )
    completion_tokens: int | None = Field(default=None, ge=0)
    encoding_name: str = ""
    safety_factor: float = Field(default=1.0, gt=0)

    # --- timing
    first_token_ms: int | None = Field(default=None, ge=0)
    total_ms: int | None = Field(default=None, ge=0)
    retries: int = Field(default=0, ge=0)
    retry_reasons: tuple[str, ...] = ()
    finish_reason: FinishReason | None = None

    # --- citations
    slots_offered: int = Field(default=0, ge=0)
    markers_seen: int = Field(default=0, ge=0)
    citations_verified: int = Field(default=0, ge=0)
    drops: tuple[CitationDrop, ...] = ()
    verification_level: Verification = Verification.RESOLVED
    verification_cache_hits: int = Field(default=0, ge=0)

    # --- policy
    redaction_scope: str = ""
    detectors_fired: dict[str, int] = Field(
        default_factory=dict,
        description="Detector name to match count. **Names, never values.**",
    )
    policy_dropped: tuple[PolicyDrop, ...] = ()

    # --- history
    turns_offered: int = Field(default=0, ge=0)
    turns_sent: int = Field(default=0, ge=0)
    history_tokens: int = Field(default=0, ge=0)
    history_conditioned_retrieval: bool = Field(
        default=False,
        description="Whether retrieval saw the conversation. It does not today — a follow-up "
        "retrieves on its own text alone — and recording it is what makes a bad follow-up "
        "diagnosable rather than mysterious.",
    )

    @property
    def drift(self) -> int | None:
        """Estimated minus true prompt tokens, or ``None`` when there is no true count."""
        if self.true_prompt_tokens is None:
            return None
        return self.estimated_prompt_tokens - self.true_prompt_tokens


__all__ = [
    "AnswerEnvelope",
    "AnswerEvent",
    "Citation",
    "CitationAccounting",
    "CitationDrop",
    "DropReason",
    "EventKind",
    "GenerationTrace",
    "PolicyDrop",
    "Verification",
]
