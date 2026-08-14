"""The query set: what gets asked, grouped by what kind of question it is.

A versioned, declared file format rather than a list of strings, for three reasons that each
correspond to a way a query set silently stops being usable.

**Provenance is required.** A set of questions someone invented to exercise the code and a set
exported from what people actually asked are not the same evidence, and by the time a number
is being quoted nobody remembers which one produced it. So every set declares which it is, the
declaration travels into every recorded preference, and a report built from an ``example`` set
says so in its first line rather than in a footnote.

**Intent is a field, not a comment.** Averaging a lookup, a how-does-this-work and an exact
identifier into one win rate hides the case that matters: a change that helps prose questions
and destroys exact-identifier lookups reads as a small win. Reporting per category is the whole
point, and it needs the categories recorded at authoring time.

**The schema is numbered and unknown versions are refused.** An export written against a later
format would otherwise load with fields quietly missing.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from manicule.evaluation.errors import QuerySetError

if TYPE_CHECKING:
    from pathlib import Path

QUERY_SET_SCHEMA_VERSION = 1
"""The format this build writes."""

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
"""What this build can read. A set rather than a maximum, so dropping support for a format is
a deliberate edit here rather than a consequence of a comparison operator."""


class Intent(StrEnum):
    """What kind of question this is, coarsely.

    Four named kinds plus an escape, chosen because they fail *differently* rather than because
    they are a taxonomy. A lexical leg carries exact identifiers and a dense leg carries
    paraphrase; a reranker earns its cost on explanatory questions and often costs precision on
    identifier lookups. One averaged number over all four hides every one of those effects.
    """

    LOOKUP = "lookup"
    """A fact expected to sit in one place. "What port does the gateway listen on"."""

    HOW_DOES_X_WORK = "how_does_x_work"
    """An explanation, usually spanning several passages."""

    COMPARISON = "comparison"
    """Two or more things weighed against each other. Needs recall across documents."""

    EXACT_IDENTIFIER = "exact_identifier"
    """An error code, symbol, ticket number or filename. The lexical leg's home ground, and
    the category most often quietly damaged by a change that improves the average."""

    UNCATEGORIZED = "uncategorized"
    """Not yet grouped. A real state for a fresh export, and reported as its own row rather
    than folded into another category — a bucket that silently absorbs the awkward ones would
    make every other row look cleaner than it is."""


class Thumbs(StrEnum):
    """The signal a user left on the answer they got, where one was collected."""

    UP = "up"
    DOWN = "down"


class Provenance(StrEnum):
    """Where a query set came from. Required, and it travels into every recorded preference."""

    EXPORTED = "exported"
    """Taken from the queries a running system was actually asked. The expensive half of an
    evaluation set, and the only kind whose phrasing is real."""

    AUTHORED = "authored"
    """Written for this purpose. Legitimate, and worth distinguishing: authored queries tend to
    be the ones the system was built to answer."""

    EXAMPLE = "example"
    """Illustrative only. A set carrying this can demonstrate the harness end to end and can
    never produce a measured result — :attr:`QuerySet.is_evidence` is ``False``, and the report
    built from it leads with a line saying so."""


class EvalQuery(BaseModel):
    """One question, with whatever is known about it.

    Nothing here is a relevance label. The harness compares two systems on the same question
    and records which was preferred; absolute judgments are a different and much more
    expensive instrument, and building them first is how a query set never gets finished.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, description="Stable within a set. Preferences key on it.")
    text: str = Field(min_length=1)
    intent: Intent = Intent.UNCATEGORIZED
    thumbs: Thumbs | None = Field(
        default=None,
        description="What the user thought of the answer they originally got, where a running "
        "system collected it. Never used as a relevance label — it is a signal about an "
        "answer, and this harness measures retrieval.",
    )
    note: str = Field(default="", description="Anything an operator wants attached.")


class QuerySet(BaseModel):
    """A named, versioned set of questions with a declared provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = QUERY_SET_SCHEMA_VERSION
    name: str = Field(min_length=1)
    provenance: Provenance
    description: str = ""
    exported_at: datetime | None = Field(
        default=None, description="When the source system was read. Timezone-aware or absent."
    )
    queries: tuple[EvalQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _readable_and_consistent(self) -> Self:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(str(version) for version in sorted(SUPPORTED_SCHEMA_VERSIONS))
            msg = (
                f"query set schema version {self.schema_version} is not one this build reads "
                f"({supported}). A newer format loaded by an older build would lose fields "
                f"silently, so it is refused instead."
            )
            raise ValueError(msg)
        seen: set[str] = set()
        repeated: set[str] = set()
        for query in self.queries:
            if query.id in seen:
                repeated.add(query.id)
            seen.add(query.id)
        if repeated:
            duplicates = sorted(repeated)
            msg = (
                f"duplicate query ids: {', '.join(duplicates)}. Preferences are keyed by id, so "
                f"a repeat would overwrite an earlier judgment rather than add to it."
            )
            raise ValueError(msg)
        if self.exported_at is not None and self.exported_at.tzinfo is None:
            msg = "exported_at must be timezone-aware; a naive timestamp has no defined meaning"
            raise ValueError(msg)
        return self

    @property
    def is_evidence(self) -> bool:
        """Whether results from this set may be described as measured.

        ``False`` for an example set, and the report reads this rather than deciding for
        itself. The alternative — a convention that example sets get labeled by whoever
        publishes the number — is the convention that fails.
        """
        return self.provenance is not Provenance.EXAMPLE

    def by_intent(self) -> dict[Intent, tuple[EvalQuery, ...]]:
        """The queries grouped by category, in declaration order within each.

        Only categories that occur. An empty row for a category nobody wrote queries for would
        report a coverage gap as a result.
        """
        grouped: dict[Intent, list[EvalQuery]] = {}
        for query in self.queries:
            grouped.setdefault(query.intent, []).append(query)
        return {intent: tuple(queries) for intent, queries in grouped.items()}


def load_query_set(path: Path) -> QuerySet:
    """Read a query set from JSON.

    Raises:
        QuerySetError: The file is missing, is not JSON, or is not a query set. One error type
            for all three, because the caller's response is the same and the message carries
            the difference.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"could not read query set {path}: {error}"
        raise QuerySetError(msg) from error
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as error:
        msg = f"{path} is not valid JSON: {error}"
        raise QuerySetError(msg) from error
    try:
        return QuerySet.model_validate(payload)
    except ValidationError as error:
        msg = (
            f"{path} is not a query set this build can read: {error}. "
            f"The format is documented in docs/evaluation.md §2."
        )
        raise QuerySetError(msg) from error


def dump_query_set(query_set: QuerySet, path: Path) -> None:
    """Write a query set as JSON, so an export script has a writer rather than a format to copy.

    Exporting from a running system is an operator's step and deliberately not part of this
    package — it needs credentials and a schema this project does not own. What is shipped is
    the target: build :class:`EvalQuery` values from whatever the export produced, declare the
    provenance, and call this.
    """
    path.write_text(
        json.dumps(query_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "QUERY_SET_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "EvalQuery",
    "Intent",
    "Provenance",
    "QuerySet",
    "Thumbs",
    "dump_query_set",
    "load_query_set",
]
