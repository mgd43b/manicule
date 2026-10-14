"""A structured memory: what a piece of knowledge claims, and what it rests on.

A memory about how an environment actually works — *"failover to the second gateway is manual
since the certificate rotation"* — is a claim somebody made, in a place, at a time. Without the
place and the time it is folklore: it cannot be checked, it cannot be dated, and nothing
distinguishes it from the thing somebody wished were true. This module is the record of those
facts, kept beside the memory's text rather than inside it.

**What is here and what is in the body.** The prose a reader searches for — the title, the
problem, the answer and where it applies — is the *body* of the file, because the body is what
is indexed and what a citation can point into. What is here is what a reader sees at a glance,
dates and weighs: the kind of knowledge it is, what it rests on, how well, and how it relates to
other memories. Front matter is not indexed (:mod:`manicule.parsers.front_matter`), so a field
that has to be retrieved by its words does not belong in it.

**Evidence is attested, never verified.** Manicule cannot open a Slack permalink, and nothing in
this module claims it did. :class:`EvidenceStatus` says how well-supported the *author* found
the claim, and every value of it is a statement about the sources rather than a check of them.
The word "verified" appears nowhere in this vocabulary on purpose: the citations an answer
carries are verified against the document they quote (``Verification``), and a second use of the
word for something that is not that would let one be read as the other.

**The quote is required.** A permalink is a pointer, and pointers die — a retention policy deletes
the message, a channel is archived, a workspace is migrated. The excerpt is what survives, so a
source with no quote is a citation that expires. The cost is copying text out of wherever it was
said into a corpus every member can search, which is a decision the operator makes by choosing
what to author, not one this module can make for them.

**Three timestamps, never folded together.** :attr:`EvidenceSource.said_at` is when the *source*
said it. The memory's own file carries when it was written down (git has that), and
``documents.indexed_at`` is when manicule read it. A memory that looks fresh because somebody
re-ran an import is the failure the split exists to prevent, so
:attr:`KnowledgeRecord.newest_said_at` is the only age this module reports.

**Where the record lives.** In ``documents.metadata`` under :data:`KNOWLEDGE_KEY`, read back
through :attr:`manicule.core.content.Document.knowledge`, for the reason
:mod:`manicule.core.provenance` gives: nothing queries it, so it is not a column. It is
**validated on every read**, because it originates in a file in the corpus and has been through
a database since anyone last looked at it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Final, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from manicule.core.provenance import require_aware, require_citable_uri, require_printable

KNOWLEDGE_KEY: Final = "structured_memory"
"""The ``documents.metadata`` key the record is stored under.

One name, exported: two spellings of it are two records, and a hit would show whichever one it
happened to look for.
"""

MARKER: Final = "knowledge_type"
"""The front-matter key whose presence says this document is a structured memory.

A marker rather than a switch, so no installation has to turn anything on and an ordinary
Markdown corpus — whose front matter has never held this key — is never read as one.
"""

FIELDS: Final = (MARKER, "evidence_status", "evidence", "relationships")
"""Every front-matter key a structured memory reads, in the order the format documents them."""

MAX_EVIDENCE: Final = 32
"""Most sources one memory may cite.

A bound on the record rather than on any one field: these are rendered into every hit that
returns the memory, so a few hundred would turn a ranked list into one document's bibliography.
"""

MAX_RELATIONSHIPS: Final = 32

CORROBORATION: Final = 2
"""How many distinct places the evidence has to point at before it is corroboration.

Two, and the word *distinct* is the rule: the same message cited twice is one source said once.
"""

MAX_QUOTE_CHARS: Final = 2000
"""Longest excerpt one source may carry.

Past this a "quote" is a copy of the thread, and what the reader needed was the sentence that
supports the claim. Refused rather than truncated, because a truncated quotation is a different
quotation presented as the source's own.
"""

_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

KIND_EXAMPLES: Final = ("slack", "jira", "confluence", "email", "meeting", "incident", "web")
"""What :attr:`EvidenceSource.kind` is usually. Examples and not an allowlist: the field is
source-agnostic by design, and an enumeration of products would be a list somebody edits every
time a team adopts a tool."""


class KnowledgeType(StrEnum):
    """What sort of knowledge a memory holds. Decides how a reader should act on it."""

    PROCEDURE = "procedure"
    """How to do something, in steps."""

    CONSTRAINT = "constraint"
    """A limit or a rule: what must, must not, or cannot be done."""

    FACT = "fact"
    """How something is — an address, a version, a topology."""

    DECISION = "decision"
    """What was chosen, and why."""

    KNOWN_ISSUE = "known_issue"
    """A failure mode, and what to do about it until it is fixed."""

    OWNERSHIP = "ownership"
    """Who owns something, and where to take a question about it."""


class EvidenceStatus(StrEnum):
    """How well the sources support the claim, as the author found them.

    Every value is a statement about the *sources*. None says manicule checked anything.
    """

    SINGLE_SOURCE = "single_source"
    """One source says it."""

    CORROBORATED = "corroborated"
    """Two or more independent sources say it."""

    DISPUTED = "disputed"
    """A source, or another memory, says otherwise."""

    SUPERSEDED = "superseded"
    """Replaced by a newer memory; the replacement is named in the relationships."""


class RelationKind(StrEnum):
    """How one memory stands to another."""

    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    RELATES_TO = "relates_to"


class KnowledgeError(ValueError):
    """A declared record that cannot be stored, with a sentence for whoever wrote it."""


class EvidenceSource(BaseModel):
    """One place the claim was said, when, and what exactly was said.

    Every field is the source's own, or empty. In particular ``said_at`` is when the *source*
    said it, never when somebody copied it here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(
        strict=True,
        description="What sort of place: slack, jira, confluence, email, meeting, incident, "
        "web. Lowercase, one token.",
    )
    locator: str = Field(
        strict=True, description="Where a reader goes to see it. An http or https address."
    )
    said_at: datetime = Field(
        description="When the source said it, as a full timestamp with a UTC offset."
    )
    quote: str = Field(
        strict=True,
        description="What the source said, verbatim, and the sentence that supports "
        "the claim rather than the thread around it.",
    )
    author: str = Field(
        default="", strict=True, description="Who said it, as the source names them. Optional."
    )

    @field_validator("said_at", mode="before")
    @classmethod
    def _timestamp(cls, value: object) -> object:
        """Refuse anything that is not already a timestamp or the text of one.

        Pydantic would read an integer as seconds since 1970 and a bare date as midnight, and
        both are a claim about *when* that the author never made. Midnight UTC is a time; a
        date is not.
        """
        if isinstance(value, datetime | str):
            return value
        msg = (
            "must be a full timestamp with a UTC offset, like 2026-03-02T14:05:00Z — a date "
            "alone or a number says less than a source's own time does"
        )
        raise ValueError(msg)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not _KIND.fullmatch(self.kind):
            msg = (
                f"kind {self.kind!r} must be one lowercase token of letters, digits, '_' or '-' "
                f"(for example {', '.join(KIND_EXAMPLES)})"
            )
            raise ValueError(msg)
        require_citable_uri(
            self.locator,
            field="locator",
            hint="Cite the permalink, not a path on a machine: a reader has to be able to open it.",
        )
        require_aware(self.said_at, field="said_at")
        if not self.quote.strip():
            msg = (
                "quote is empty. The excerpt is what survives when the source is deleted, so "
                "every source carries the words it was cited for"
            )
            raise ValueError(msg)
        require_printable(self.quote, field="quote", limit=MAX_QUOTE_CHARS, allow="\n\t")
        require_printable(self.author, field="author")
        return self


class Relationship(BaseModel):
    """A typed pointer to another memory, by its slug."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relation: RelationKind
    target: str = Field(strict=True, description="The other memory's slug.")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not _SLUG.fullmatch(self.target):
            msg = (
                f"target {self.target!r} is not a slug: letters, digits, '_' and '-' only, "
                f"without a path or a '.md' suffix"
            )
            raise ValueError(msg)
        return self


class KnowledgeRecord(BaseModel):
    """What a memory claims to be, and what it rests on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    knowledge_type: KnowledgeType
    evidence: tuple[EvidenceSource, ...] = Field(min_length=1, max_length=MAX_EVIDENCE)
    evidence_status: EvidenceStatus
    relationships: tuple[Relationship, ...] = Field(default=(), max_length=MAX_RELATIONSHIPS)

    @property
    def oldest_said_at(self) -> datetime:
        """When the earliest source said it."""
        return min(source.said_at for source in self.evidence)

    @property
    def newest_said_at(self) -> datetime:
        """When the latest source said it — the only age this record reports.

        It dates the *evidence*, which is what a reader deciding whether to trust an
        operational fact needs: a memory written last week from a message sent three years ago
        is three years old.
        """
        return max(source.said_at for source in self.evidence)

    @property
    def independent_sources(self) -> int:
        """How many distinct places the evidence points at."""
        return len({source.locator for source in self.evidence})

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        independent = self.independent_sources
        if self.evidence_status is EvidenceStatus.CORROBORATED and independent < CORROBORATION:
            msg = (
                "evidence_status is corroborated and the evidence points at one place. "
                "Corroboration is two sources that are not the same message — cite the second, "
                "or say single_source"
            )
            raise ValueError(msg)
        if self.evidence_status is EvidenceStatus.SINGLE_SOURCE and independent > 1:
            msg = (
                f"evidence_status is single_source and the evidence points at {independent} "
                f"places. Say corroborated, or drop the extra sources"
            )
            raise ValueError(msg)
        if self.evidence_status is EvidenceStatus.SUPERSEDED and not any(
            link.relation is RelationKind.SUPERSEDED_BY for link in self.relationships
        ):
            msg = (
                "evidence_status is superseded and nothing says by what. Add a relationship "
                "with relation: superseded_by and the replacement's slug"
            )
            raise ValueError(msg)
        seen = {(link.relation, link.target) for link in self.relationships}
        if len(seen) != len(self.relationships):
            msg = "relationships names the same relation to the same memory more than once"
            raise ValueError(msg)
        return self

    @classmethod
    def from_declared(cls, declared: Mapping[str, object]) -> Self:
        """The record a front-matter mapping declares, with ``evidence_status`` derived if absent.

        Derived rather than required: whether two sources are the same message is arithmetic,
        and an author asked to state a fact the evidence already shows has one more way to be
        wrong. ``disputed`` and ``superseded`` cannot be derived from the evidence, so those
        are declared.

        Raises:
            KnowledgeError: Naming every field that was wrong and what would fix it.
        """
        fields = dict(declared)
        if "evidence_status" not in fields:
            fields["evidence_status"] = _derived_status(fields.get("evidence"))
        try:
            return cls.model_validate(fields)
        except ValidationError as exc:
            raise KnowledgeError(_describe(exc)) from exc


def _derived_status(evidence: object) -> str:
    """``single_source`` or ``corroborated``, from how many places the evidence points at."""
    items = cast("list[object]", evidence) if isinstance(evidence, list) else []
    places = {
        cast("Mapping[str, object]", item).get("locator")
        for item in items
        if isinstance(item, Mapping)
    }
    # The same count `KnowledgeRecord.independent_sources` makes, over input that has not been
    # validated yet; the model's own check then holds a declared status to it.
    return (
        EvidenceStatus.CORROBORATED.value
        if len(places) >= CORROBORATION
        else EvidenceStatus.SINGLE_SOURCE.value
    )


MAX_ECHO_CHARS: Final = 64
"""Longest author-supplied value a refusal repeats back.

Only a short plain string: the value could be a multi-megabyte scalar or a YAML alias expanding
to a structure whose ``repr`` is exponential, and a refusal must cost what the input cost to
parse and no more.
"""

MAX_ERRORS_REPORTED: Final = 8
MAX_REASON_CHARS: Final = 1000


def bounded_reason(reason: str, *, more: int = 0) -> str:
    """``reason`` held to :data:`MAX_REASON_CHARS`, saying so when anything was left out.

    A stored reason travels with every hit, citation and listing that returns the memory, so it
    is a diagnostic and not a transcript. Truncating a *diagnostic* is safe where truncating
    evidence is not: the author still has the file, and the first errors are the ones to fix.
    """
    suffix = f" (and {more - MAX_ERRORS_REPORTED} more)" if more > MAX_ERRORS_REPORTED else ""
    if len(reason) > MAX_REASON_CHARS:
        reason = reason[: MAX_REASON_CHARS - 1] + "…"
    return reason + suffix


def normalize_slug(slug: str) -> str:
    """A slug as memories link to each other: case folded, hyphen and underscore one separator."""
    return re.sub(r"[-_]+", "-", slug.strip().lower())


def _describe(error: ValidationError) -> str:
    """Pydantic's errors as the one-line-per-field sentence a caller can act on."""
    lines: list[str] = []
    for detail in error.errors():
        path = ""
        for part in detail["loc"]:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        message = str(detail["msg"]).removeprefix("Value error, ")
        if detail["type"] == "extra_forbidden":
            allowed = ", ".join(_allowed_beside(detail["loc"]))
            message = f"is not a field here; the fields are {allowed}"
        elif detail["type"] == "enum":
            # The list of what is accepted does not say which of the author's words was not, and
            # an enum value is a short token the author chose — never a quote or a locator.
            echoed = detail["input"]
            if isinstance(echoed, str) and len(echoed) <= MAX_ECHO_CHARS:
                message = f"{message} (got {echoed!r})"
        lines.append(f"{path}: {message}" if path else message)
    return bounded_reason("; ".join(lines[:MAX_ERRORS_REPORTED]), more=len(lines))


def _allowed_beside(location: tuple[int | str, ...]) -> tuple[str, ...]:
    """The field names the unknown field's neighbors have, for the refusal to name."""
    if "evidence" in location:
        return tuple(EvidenceSource.model_fields)
    if "relationships" in location:
        return tuple(Relationship.model_fields)
    return FIELDS


class Knowledge(BaseModel):
    """What is stored for one document: a record, or the reason there is not one.

    **A refusal is stored rather than dropped**, the way
    :attr:`~manicule.core.provenance.Provenance.unavailable_reason` is, because a document that
    declared evidence and silently shows none presents as a memory nobody cited — which is
    indistinguishable from one that never tried.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: KnowledgeRecord | None = None
    unavailable_reason: str = ""

    @model_validator(mode="after")
    def _one_of(self) -> Self:
        if (self.record is None) == (not self.unavailable_reason):
            msg = "a stored memory record carries either the record or the reason it is absent"
            raise ValueError(msg)
        return self

    def as_metadata_value(self) -> dict[str, JsonValue]:
        """This record, JSON-shaped, for storing under :data:`KNOWLEDGE_KEY`."""
        return cast("dict[str, JsonValue]", self.model_dump(mode="json"))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Knowledge | None:
        """The record in ``metadata``, validated, or ``None`` when there is not a usable one.

        **Failing closed is the point**, for the reason
        :meth:`manicule.core.provenance.Provenance.from_metadata` gives: one malformed row must
        not break every listing that touches it, and a stored value that no longer validates is
        input again.
        """
        raw = metadata.get(KNOWLEDGE_KEY)
        if not isinstance(raw, Mapping):
            return None
        try:
            return cls.model_validate(dict(cast("Mapping[str, object]", raw)))
        except ValueError:
            return None


__all__ = [
    "CORROBORATION",
    "FIELDS",
    "KIND_EXAMPLES",
    "KNOWLEDGE_KEY",
    "MARKER",
    "MAX_ECHO_CHARS",
    "MAX_ERRORS_REPORTED",
    "MAX_EVIDENCE",
    "MAX_QUOTE_CHARS",
    "MAX_REASON_CHARS",
    "MAX_RELATIONSHIPS",
    "EvidenceSource",
    "EvidenceStatus",
    "Knowledge",
    "KnowledgeError",
    "KnowledgeRecord",
    "KnowledgeType",
    "RelationKind",
    "Relationship",
    "bounded_reason",
    "normalize_slug",
]
