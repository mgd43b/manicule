"""Conversation memory: whole turns, in pairs, newest first.

Three rules do most of the work, and each of them is a refusal of something that looks
cheaper:

**Whole turns only.** A half message misrepresents what was said, in the same way a trimmed
passage misrepresents a source. If a turn does not fit, it is dropped.

**Turns are dropped in user/assistant pairs.** Keeping an assistant turn whose question is
gone leaves the model an answer to something it cannot see, which is worse than having
neither — it invites the model to infer the missing question.

**No summarisation.** A rolling summary is a generated artefact that then gets treated as a
record of what was said, and it costs a model call per turn.

The current user turn is never dropped: it is the question. If it alone does not fit, that is
a refusal with the numbers named, not a truncation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from manicule.generation.answers import Citation
from manicule.generation.budget import TokenEstimator
from manicule.generation.markers import ATTEMPT_PREFIX, MARKER_CLOSE, escape_markers
from manicule.generation.prompt import ChatMessage

_MARKER = re.compile(
    rf"{re.escape(ATTEMPT_PREFIX)}\s*:\s*(\d+(?:\s*,\s*\d+)*)\s*{re.escape(MARKER_CLOSE)}",
    re.IGNORECASE,
)
"""Markers in a stored answer, for neutralisation. Whole-string rather than incremental:
history is complete text, not a stream."""


@dataclass(frozen=True, slots=True)
class Turn:
    """One stored message, with the citations that were verified for it."""

    role: Literal["user", "assistant"]
    content: str
    citations: tuple[Citation, ...] = ()

    def citation_for(self, slot: int) -> Citation | None:
        for citation in self.citations:
            if citation.slot == slot:
                return citation
        return None


def neutralise_markers(turn: Turn) -> str:
    """Rewrite a stored answer's markers into a non-bindable textual reference.

    **Slot numbers are per-answer.** Turn 1's ``[[cite:3]]`` referred to turn 1's third
    passage; turn 4 has an entirely different context and its slot 3 is a different document.
    Feeding turn 1's answer back verbatim hands the model marker syntax that binds to
    something else — and the model copies the pattern.

    So markers become ``[cited: 'Deploy runbook' > Rollback]``, carrying enough for the model
    to follow the conversation and nothing the binder will act on. They are never re-verified
    and never re-bound: the citation records for prior turns already exist and are what a
    reader sees.

    This is the one place answer text is transformed, and it does not contradict the drop
    rule: like redaction, it applies to **a copy on its way into a prompt**. The stored answer
    is untouched and nothing a reader sees changes.
    """

    def replace(match: re.Match[str]) -> str:
        labels = [
            found.label
            for found in (turn.citation_for(int(slot)) for slot in match.group(1).split(","))
            if found is not None
        ]
        return f"[cited: {'; '.join(labels)}]" if labels else "[cited]"

    return _MARKER.sub(replace, turn.content)


def as_message(turn: Turn) -> ChatMessage:
    """One turn as the provider will see it, with markers neutralised and syntax escaped.

    ``escape_markers`` runs afterwards for the same reason it runs on a passage: a *user*
    turn can contain marker syntax the model would otherwise copy, and a user can paste
    anything.
    """
    content = neutralise_markers(turn) if turn.role == "assistant" else turn.content
    return {"role": turn.role, "content": escape_markers(content)}


@dataclass(frozen=True, slots=True)
class HistoryPlan:
    """Which turns are being sent, and what they cost."""

    messages: tuple[ChatMessage, ...]
    turns_offered: int
    turns_sent: int
    tokens: int

    @property
    def turns_dropped(self) -> int:
        return self.turns_offered - self.turns_sent


def fit_history(turns: Sequence[Turn], *, budget: int, estimator: TokenEstimator) -> HistoryPlan:
    """Fit as many whole, paired turns as ``budget`` allows, newest first.

    Measured with the **generation** model's tokenizer, never the embedder's — the two count
    different things for different models, and using one for the other is the category error
    that budget arithmetic exists to avoid.

    ``history_tokens`` is a separate budget from ``context_tokens`` and neither lends to the
    other. A shared pool sounds strictly better and is not: a long conversation would starve
    retrieval, so the tenth turn of a chat gets fewer passages than the first for the same
    question, and the answer gets worse for a reason invisible to everybody.
    """
    pairs = _pairs(turns)
    kept: list[tuple[ChatMessage, ...]] = []
    spent = 0
    for pair in reversed(pairs):
        rendered = tuple(as_message(turn) for turn in pair)
        cost = estimator.count_all(message["content"] for message in rendered)
        if spent + cost > budget:
            # Older turns cannot fit either, and stopping keeps the kept set contiguous —
            # a conversation with a hole in the middle reads as two conversations.
            break
        kept.append(rendered)
        spent += cost
    kept.reverse()
    messages = tuple(message for pair in kept for message in pair)
    return HistoryPlan(
        messages=messages,
        turns_offered=len(turns),
        turns_sent=sum(len(pair) for pair in kept),
        tokens=spent,
    )


def _pairs(turns: Sequence[Turn]) -> list[tuple[Turn, ...]]:
    """Group a transcript into user/assistant pairs, in order.

    A leading assistant turn, or two user turns in a row, is not an error — a conversation
    can be repaired, imported or interrupted — so an unpaired turn becomes a group of one
    rather than being discarded. What the grouping guarantees is that an assistant turn is
    never sent without the user turn it answers.
    """
    grouped: list[tuple[Turn, ...]] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        following = turns[index + 1] if index + 1 < len(turns) else None
        if turn.role == "user" and following is not None and following.role == "assistant":
            grouped.append((turn, following))
            index += 2
        else:
            grouped.append((turn,))
            index += 1
    return grouped


__all__ = ["HistoryPlan", "Turn", "as_message", "fit_history", "neutralise_markers"]
