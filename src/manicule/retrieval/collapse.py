"""Near-duplicate collapse within a conversation: a passage a thread repeats is shown once.

A mail message keeps the reply chain it quotes, so a thread of thirty messages carries its first
message thirty times. Ranked per chunk, a question about that message fills the page with copies
of one paragraph from different replies, and the document that answered it differently — the one
a person most needs to see beside it — is pushed off the end. ``docs/connectors/gmail.md`` §8 has
the measurement.

**The rule.** Walk the ranking best first. A candidate whose document names a conversation is
dropped when at least ``containment`` of its word trigrams already appear in candidates kept
above it from the same conversation — unless what differs is a few words changed inside text
shown above, which is a correction rather than a copy. Nothing is reordered and nothing is
rescored, so what survives keeps the rank fusion gave it; the stage only removes what would
show a person text they have already been shown.

**Whichever copy ranked first is the one kept, so the original can be the one dropped.** When a
reply quoting a message outranks it, the message's own chunk is a repeat of what was shown and
goes, and the passage is then cited to the reply that quoted it. Protecting originals was tried
and measured: a message's opening chunk holds its own new text beside the history it quotes, so
every rule that kept originals also kept most of the copies. ``docs/connectors/gmail.md`` §8 has
the numbers for both.

**Within a conversation only.** Two documents from different threads are never compared,
however alike their text, because a repeat there is not a quotation and may be the very
difference a question is asking about. A document with no conversation is passed through
untouched, which makes the stage a no-op for every source that does not record one.

**Quote markers are not text.** A plain-text reply prefixes the quoted chain with ``>``, one per
level, and an HTML-only reply arrives with none at all once converted. Stripping the markers
before comparing is what makes the two forms of one quotation the same text, and what keeps a
passage quoted three levels deep recognizable as the one quoted once.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from manicule.retrieval.config import CollapseConfig
from manicule.retrieval.prefilter import join_filter

if TYPE_CHECKING:
    from manicule.core.content import Metadata
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Candidate, Query

SHINGLE = 3
"""Words per shingle. Long enough that two passages sharing a phrase are not the same passage,
short enough that one wrapped differently still is."""

_QUOTE_MARKERS = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.MULTILINE)
_WORD = re.compile(r"\w+")
_HEADER_LINE = re.compile(
    r"^[ \t]*(?:>[ \t]?)*(?:from|to|cc|bcc|date|sent|subject):.*$", re.IGNORECASE | re.MULTILINE
)


def trigrams(text: str) -> list[tuple[str, ...]]:
    """The word trigrams of ``text`` in order, with quote markers, case and whitespace folded."""
    words = _WORD.findall(_QUOTE_MARKERS.sub("", text).casefold())
    if len(words) < SHINGLE:
        return [tuple(words)] if words else []
    return [tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def body_trigrams(text: str) -> list[tuple[str, ...]]:
    """:func:`trigrams` with header lines — ``From:``, ``To:``, ``Date:`` and the like — removed.

    Two messages' headers differ the way a correction does, a few words apart inside text that
    is otherwise the same, so they are left out of the search for corrections. They still count
    towards how much of a candidate was shown.
    """
    return trigrams(_HEADER_LINE.sub("", text))


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """The word trigrams of ``text`` as a set, folded as :func:`trigrams` folds them."""
    return frozenset(trigrams(text))


EDIT = 3
"""The most words a correction changes, inserts or removes in one place, for it to be told
from new text: past that, what differs is a passage of its own, and ``containment`` decides."""


class _Shown:
    """The text kept above, in one conversation: its trigrams, and where each stood."""

    def __init__(self) -> None:
        self.trigrams: set[tuple[str, ...]] = set()
        self._at: list[dict[tuple[str, ...], list[int]]] = []

    def add(self, ordered: list[tuple[str, ...]], body: list[tuple[str, ...]]) -> None:
        """Record a kept candidate: all of its trigrams, and where its body's stood."""
        self.trigrams.update(ordered)
        positions: dict[tuple[str, ...], list[int]] = {}
        for index, trigram in enumerate(body):
            positions.setdefault(trigram, []).append(index)
        self._at.append(positions)

    def edits(self, ordered: list[tuple[str, ...]]) -> bool:
        """Whether ``ordered`` is shown text with a few words changed in the middle.

        A correction leaves the words either side of it in place, so its signature is a short
        run of unshown trigrams whose neighbors — the shown trigram just before and just after —
        stand in one chunk above, at most :data:`EDIT` words apart, with at most :data:`EDIT`
        words between them here. A seam between two passages shown in different chunks, and a
        passage no chunk above showed, have no such neighbors, so neither is mistaken for one.
        """
        covered = [trigram in self.trigrams for trigram in ordered]
        index = 0
        while index < len(ordered):
            if covered[index]:
                index += 1
                continue
            start = index
            while index < len(ordered) and not covered[index]:
                index += 1
            if start == 0 or index == len(ordered):
                continue  # unshown text at an end is where this chunk began or ended
            # The neighbors end and begin ``run - 2`` words apart here: none for a removed
            # word, one for a changed or inserted one.
            between = index - start - 2
            if between <= EDIT and self._near(ordered[start - 1], ordered[index]):
                return True
        return False

    def _near(self, before: tuple[str, ...], after: tuple[str, ...]) -> bool:
        """Whether one chunk above holds ``before`` and then ``after``, a few words apart."""
        for positions in self._at:
            for left in positions.get(before, ()):
                for right in positions.get(after, ()):
                    if 0 <= right - left - SHINGLE <= EDIT:
                        return True
        return False


class CollapseStage:
    """Drop a candidate that repeats text already ranked above it in the same conversation."""

    def __init__(
        self, *, docstore: DocStore, config: CollapseConfig | None = None, name: str = "collapse"
    ) -> None:
        self.name = name
        self._docstore = docstore
        self._config = config or CollapseConfig()

    def describe(self) -> Metadata:
        """Which field groups a conversation, and how much repetition drops a candidate."""
        return dict(self._config.model_dump(mode="json"))

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        conversations = await self._conversations(query, candidates)
        if not conversations:
            return list(candidates)
        shown: dict[tuple[str, str], _Shown] = {}
        kept: list[Candidate] = []
        for candidate in candidates:
            conversation = conversations.get(candidate.chunk.document_id)
            if conversation is None:
                kept.append(candidate)
                continue
            text = candidate.chunk.text
            ordered, body = trigrams(text), body_trigrams(text)
            above = shown.setdefault(conversation, _Shown())
            if self._repeats(ordered, body, above):
                continue
            above.add(ordered, body)
            kept.append(candidate)
        return kept

    def _repeats(
        self, ordered: list[tuple[str, ...]], body: list[tuple[str, ...]], above: _Shown
    ) -> bool:
        """Whether this text was already shown, and is not a correction of what was.

        Enough of it must appear above — ``containment`` of its distinct trigrams — and none of
        what does not may be an edit (:meth:`_Shown.edits`).
        """
        own = set(ordered)
        if not own or len(own & above.trigrams) < self._config.containment * len(own):
            return False
        return not above.edits(body)

    async def _conversations(
        self, query: Query, candidates: list[Candidate]
    ) -> dict[str, tuple[str, str]]:
        """The conversation each candidate's document belongs to, where it names one.

        Keyed by source as well as by the field's value: a thread id is unique within the
        mailbox that issued it, and two sources in one workspace can issue the same one.

        Read under the document-level half of the query's filter, narrowed to these documents,
        so this is no wider a read than the one that produced the candidates. Only that half:
        ``kinds`` and ``langs`` restrict chunks, which a list of documents cannot apply and the
        store refuses, and the candidates already satisfy them.
        """
        scope = join_filter(query.filter)
        wanted = frozenset(candidate.chunk.document_id for candidate in candidates)
        if scope.document_ids:
            wanted &= scope.document_ids
        if not wanted:
            return {}
        documents = await self._docstore.list_documents(
            scope.model_copy(update={"document_ids": wanted}), limit=len(wanted)
        )
        key = self._config.thread_key
        return {
            document.id: (document.source, str(document.metadata[key]))
            for document in documents
            if document.metadata.get(key) not in (None, "")
        }


__all__ = ["EDIT", "SHINGLE", "CollapseStage", "body_trigrams", "shingles", "trigrams"]
