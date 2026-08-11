"""Who decides, and what they are shown.

A judge sees two anonymous ranked lists and picks one, ties them, or says neither answered.
Seconds per query is the design constraint, and everything here follows from it: no scale, no
per-passage rating, one keypress.

Nothing in this module knows about a terminal library. :class:`StreamJudge` reads and writes
plain text streams, so the same code path that a person drives interactively is the one the
tests drive with a string — a judging loop that can only be exercised by a human is a judging
loop nobody has exercised.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.evaluation.errors import EvaluationError
from manicule.evaluation.preference import Pairing, Preference, Slot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

    from manicule.evaluation.systems import ResultItem

SNIPPET = 220
"""Characters of each passage shown. Enough to judge relevance, short enough to read eight of
them without scrolling."""

SHOWN = 5
"""Results shown per side. Judging twenty passages a query is how a session takes an hour and
never gets repeated."""


class JudgingStoppedError(EvaluationError):
    """The judge asked to stop. Everything already recorded stays recorded."""


@runtime_checkable
class Judge(Protocol):
    """Something that expresses a preference between two anonymous result sets."""

    @property
    def label(self) -> str:
        """Who or what this is. Recorded on every judgement it makes."""
        ...

    async def judge(self, pairing: Pairing) -> tuple[Preference, str] | None:
        """Decide, or return ``None`` to skip this query without recording anything.

        Raises:
            JudgingStoppedError: End the session early.
        """
        ...


class ScriptedJudge:
    """A judge that replays a decision per query id.

    For tests and for re-running a recorded session against a changed pipeline. A query it has
    no answer for is skipped rather than defaulted: a default would be a judgement nobody made.
    """

    def __init__(self, decisions: Mapping[str, Preference], *, label: str = "scripted") -> None:
        self._decisions = dict(decisions)
        self._label = label
        self.seen: list[str] = []

    @property
    def label(self) -> str:
        return self._label

    async def judge(self, pairing: Pairing) -> tuple[Preference, str] | None:
        self.seen.append(pairing.query.id)
        decision = self._decisions.get(pairing.query.id)
        return None if decision is None else (decision, "")


class SlotJudge:
    """A judge that always picks the same slot, whatever is in it.

    Not a convenience. It is how the blinding is shown to work: because slot assignment is a
    keyed hash of the query id rather than a fixed order, a judge with a pure position bias
    produces a split indistinguishable from a coin instead of a clean win for whichever system
    was passed as ``left``. Without blinding this judge would score 100% for one side, and that
    is a number a harness must never be able to produce.
    """

    def __init__(self, slot: Slot = Slot.A, *, label: str = "position-bias") -> None:
        self._slot = slot
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    async def judge(self, pairing: Pairing) -> tuple[Preference, str] | None:
        return pairing.resolve(self._slot), f"always chose slot {self._slot.value}"


SLOT_KEYS = {"a": Slot.A, "b": Slot.B}
"""Keypresses that name a *slot*, which must be resolved through the pairing.

Kept in a separate table from :data:`VERDICT_KEYS` rather than one map of mixed meanings.
Pressing ``a`` does not mean "prefer the left system" — it means "prefer whatever was shown
first", and which system that was is a keyed hash of the query id. A single table returning a
:class:`~manicule.evaluation.preference.Preference` for ``a`` would invite exactly the
one-line simplification that discards the blinding, and the resulting bias would be invisible
in every report it produced.
"""

VERDICT_KEYS = {"t": Preference.TIE, "n": Preference.NEITHER}
"""Keypresses that name an outcome directly. Neither names a side, so neither needs resolving."""

PROMPT = "[a] A better  [b] B better  [t] tie  [n] neither  [s] skip  [q] quit > "


def render_pairing(pairing: Pairing, *, shown: int = SHOWN, snippet: int = SNIPPET) -> str:
    """The two result sets, side by side and unlabelled.

    Nothing identifying either system appears — not the configuration label, not the latency,
    not the scores. A judge who can tell which list came from the system they built is not
    blind, and the preference then measures the judge.
    """
    lines = [
        "=" * 100,
        f"Q: {pairing.query.text}",
        f"   intent: {pairing.query.intent.value}   id: {pairing.query.id}",
    ]
    for slot in (Slot.A, Slot.B):
        lines.append("")
        lines.append(f"--- {slot.value} " + "-" * 94)
        lines.extend(_render_items(pairing.shown_as(slot).items[:shown], snippet=snippet))
    lines.append("")
    return "\n".join(lines)


def _render_items(items: Sequence[ResultItem], *, snippet: int) -> list[str]:
    if not items:
        return ["    (nothing returned)"]
    rendered: list[str] = []
    for rank, item in enumerate(items, start=1):
        head = item.title or item.location or item.document_id
        body = textwrap.shorten(" ".join(item.text.split()), width=snippet, placeholder=" …")
        rendered.append(f" {rank}. {head}")
        rendered.append(f"    {body}")
    return rendered


class StreamJudge:
    """A person, driven through two text streams.

    Streams rather than ``input()`` and ``print()`` so the loop is testable end to end, and so
    a caller can drive it from a terminal, a pipe or a transcript without this module knowing
    which.
    """

    def __init__(
        self,
        *,
        output: TextIO,
        input_stream: TextIO,
        label: str = "operator",
        shown: int = SHOWN,
        snippet: int = SNIPPET,
    ) -> None:
        self._output = output
        self._input = input_stream
        self._label = label
        self._shown = shown
        self._snippet = snippet

    @property
    def label(self) -> str:
        return self._label

    async def judge(self, pairing: Pairing) -> tuple[Preference, str] | None:
        self._output.write(render_pairing(pairing, shown=self._shown, snippet=self._snippet))
        while True:
            self._output.write(PROMPT)
            self._output.flush()
            line = self._input.readline()
            if not line:
                # End of stream. Treated as quit rather than as a skip loop, because a
                # judging session whose input has closed would otherwise spin forever.
                msg = "input ended while judging"
                raise JudgingStoppedError(msg)
            key = line.strip().casefold()
            if key in {"q", "quit"}:
                msg = "the judge asked to stop"
                raise JudgingStoppedError(msg)
            if key in {"s", "skip", ""}:
                return None
            slot = SLOT_KEYS.get(key)
            if slot is not None:
                return pairing.resolve(slot), ""
            verdict = VERDICT_KEYS.get(key)
            if verdict is not None:
                return verdict, ""
            self._output.write(f"unrecognised choice {key!r}\n")


__all__ = [
    "PROMPT",
    "SHOWN",
    "SLOT_KEYS",
    "SNIPPET",
    "VERDICT_KEYS",
    "Judge",
    "JudgingStoppedError",
    "ScriptedJudge",
    "SlotJudge",
    "StreamJudge",
    "render_pairing",
]
