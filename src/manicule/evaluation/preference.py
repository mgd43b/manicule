"""Pairwise preference: what was shown, what was chosen, and what produced each side.

Pairwise rather than absolute relevance labels, and that is the methodological choice this
package is built around. Absolute judgments need a scale, a definition of relevant, and a
minute of attention per passage; a preference between two ranked lists takes seconds and
answers the question actually being asked — *is this configuration better than the one I have*.
Absolute labels and nDCG come later, and only if preference stops discriminating between
candidate configurations. Building them first is how an evaluation set never gets finished.

**Sides are blinded.** Position bias is large and free to eliminate: which system appears as A
is decided by a keyed hash of the query id, so the assignment is deterministic, reproducible
from the recorded seed, and uncorrelated with anything about the systems. A judge who always
picks the first list therefore produces a split indistinguishable from a coin rather than a
clean win for whichever system was passed first.

**Records are append-only JSON Lines.** They are evidence. Rewriting a file of judgments in
place is how a run gets re-scored after the fact, and a line-per-record file can be appended to
across sessions, read by anything, and diffed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from manicule.evaluation.errors import PreferenceRecordError
from manicule.evaluation.probe import ProbeOutcome
from manicule.evaluation.queries import EvalQuery, Intent, Provenance, Thumbs
from manicule.evaluation.systems import SystemResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PREFERENCE_SCHEMA_VERSION = 1
"""The record format this build writes."""

SUPPORTED_PREFERENCE_VERSIONS = frozenset({1})
"""What this build can read back."""

DEFAULT_BLINDING_SEED = "manicule-evaluation"
"""The default key for side assignment.

A constant rather than a random value, so two people running the same query set against the
same systems see the same layout and can talk about record 14. Changing it reshuffles which
side is shown first and changes nothing about any result.
"""


class Side(StrEnum):
    """Which system, in the terms the harness was constructed with."""

    LEFT = "left"
    RIGHT = "right"


class Slot(StrEnum):
    """Which position a judge saw it in. The only thing a judge is shown."""

    A = "A"
    B = "B"


class Preference(StrEnum):
    """What the judge decided.

    Four outcomes, and the last two are not the same. A tie says both lists answered the
    question about equally well; ``neither`` says both failed. Collapsing them would let a
    query set on which both systems are useless read as a dead heat, which is the reading that
    stops anybody investigating.
    """

    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"
    NEITHER = "neither"


def assign_slots(query_id: str, *, seed: str = DEFAULT_BLINDING_SEED) -> tuple[Side, Side]:
    """Which side is shown as A and which as B, for this query.

    Keyed by the query id rather than by position in the run, so the layout does not change
    when a query is added, removed or reordered — and a session resumed tomorrow shows the same
    query the same way round.

    Returns:
        ``(side shown as A, side shown as B)``.
    """
    digest = hashlib.blake2b(f"{seed}\x00{query_id}".encode(), digest_size=8).digest()
    return (Side.LEFT, Side.RIGHT) if digest[0] % 2 == 0 else (Side.RIGHT, Side.LEFT)


class Pairing(BaseModel):
    """One query, both result sets, and the blinding that decides how they are shown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: EvalQuery
    left: SystemResult
    right: SystemResult
    slots: tuple[Side, Side] = Field(
        description="Which side occupies slot A and which occupies slot B."
    )
    seed: str = DEFAULT_BLINDING_SEED

    def shown_as(self, slot: Slot) -> SystemResult:
        """The result set behind a slot the judge saw."""
        side = self.slots[0] if slot is Slot.A else self.slots[1]
        return self.left if side is Side.LEFT else self.right

    def resolve(self, chosen: Slot) -> Preference:
        """Turn a slot the judge picked into a statement about a system."""
        side = self.slots[0] if chosen is Slot.A else self.slots[1]
        return Preference.LEFT if side is Side.LEFT else Preference.RIGHT

    @property
    def incomparable(self) -> tuple[str, ...]:
        """Why this pairing may not count towards a rate, labeled by side."""
        return tuple(
            f"{result.config_label}: {reason}"
            for result in (self.left, self.right)
            for reason in result.incomparable
        )


class PreferenceRecord(BaseModel):
    """One judgment, self-contained.

    Everything needed to read it back without the query set, the configuration files or the
    person who ran it: the question, both configurations in full, both corpus versions, both
    discrimination probes, and what was chosen. A record that needs context to interpret is a
    record that becomes uninterpretable the moment the context moves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PREFERENCE_SCHEMA_VERSION
    recorded_at: datetime
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    intent: Intent = Intent.UNCATEGORIZED
    thumbs: Thumbs | None = None
    query_set: str = Field(min_length=1)
    provenance: Provenance
    left: SystemResult
    right: SystemResult
    left_probe: ProbeOutcome
    right_probe: ProbeOutcome
    slots: tuple[Side, Side]
    seed: str = DEFAULT_BLINDING_SEED
    preference: Preference
    judge: str = Field(min_length=1, description="Who or what decided. Appears in the report.")
    note: str = ""

    @model_validator(mode="after")
    def _readable_and_honest(self) -> Self:
        if self.schema_version not in SUPPORTED_PREFERENCE_VERSIONS:
            supported = ", ".join(str(v) for v in sorted(SUPPORTED_PREFERENCE_VERSIONS))
            msg = (
                f"preference record schema version {self.schema_version} is not one this build "
                f"reads ({supported})"
            )
            raise ValueError(msg)
        if self.recorded_at.tzinfo is None:
            msg = "recorded_at must be timezone-aware"
            raise ValueError(msg)
        if not self.left_probe.discriminates or not self.right_probe.discriminates:
            failing = [
                probe.config_label
                for probe in (self.left_probe, self.right_probe)
                if not probe.discriminates
            ]
            msg = (
                f"a preference cannot be recorded for a side that retrieves at chance: "
                f"{', '.join(failing)}. The judgment would be a judgment about noise, and a "
                f"stored one would outlive the knowledge that it was"
            )
            raise ValueError(msg)
        return self

    @property
    def admissible(self) -> bool:
        """Whether this judgment may count towards a rate.

        A record is kept either way — it is evidence about the run — but a pairing where a run
        was degraded, cached or stopped by its own budget is not a comparison of two pipelines.
        """
        return not self.left.incomparable and not self.right.incomparable

    @property
    def inadmissible_because(self) -> tuple[str, ...]:
        return tuple(
            f"{result.config_label}: {reason}"
            for result in (self.left, self.right)
            for reason in result.incomparable
        )


def build_record(
    pairing: Pairing,
    *,
    preference: Preference,
    query_set: str,
    provenance: Provenance,
    left_probe: ProbeOutcome,
    right_probe: ProbeOutcome,
    judge: str,
    note: str = "",
) -> PreferenceRecord:
    """Freeze a judged pairing into a record."""
    return PreferenceRecord(
        recorded_at=datetime.now(UTC),
        query_id=pairing.query.id,
        query_text=pairing.query.text,
        intent=pairing.query.intent,
        thumbs=pairing.query.thumbs,
        query_set=query_set,
        provenance=provenance,
        left=pairing.left,
        right=pairing.right,
        left_probe=left_probe,
        right_probe=right_probe,
        slots=pairing.slots,
        seed=pairing.seed,
        preference=preference,
        judge=judge,
        note=note,
    )


class PreferenceStore:
    """Append-only JSON Lines on disk.

    No update and no delete, deliberately. A judgment is an observation; correcting one is
    making a new observation, and a file that can be edited in place is a file whose history
    is whatever the last writer decided it was.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: PreferenceRecord) -> None:
        """Add one judgment. Creates the file and its directory on first write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def records(self) -> Iterator[PreferenceRecord]:
        """Every judgment, in the order it was made.

        Raises:
            PreferenceRecordError: A line is not a record this build can read. Refused rather than
                skipped: a reader that drops what it does not understand reports a rate
                computed over an unknown subset.
        """
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield self._parse(line, number)

    def _parse(self, line: str, number: int) -> PreferenceRecord:
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"{self._path}:{number} is not valid JSON: {error}"
            raise PreferenceRecordError(msg) from error
        try:
            return PreferenceRecord.model_validate(payload)
        except ValidationError as error:
            msg = f"{self._path}:{number} is not a preference record this build reads: {error}"
            raise PreferenceRecordError(msg) from error

    def __len__(self) -> int:
        return sum(1 for _ in self.records())


__all__ = [
    "DEFAULT_BLINDING_SEED",
    "PREFERENCE_SCHEMA_VERSION",
    "SUPPORTED_PREFERENCE_VERSIONS",
    "Pairing",
    "Preference",
    "PreferenceRecord",
    "PreferenceStore",
    "Side",
    "Slot",
    "assign_slots",
    "build_record",
]
