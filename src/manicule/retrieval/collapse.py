"""Near-duplicate collapse within a conversation: a passage a thread repeats is shown once.

A mail message keeps the reply chain it quotes, so a thread of thirty messages carries its first
message thirty times. Ranked per chunk, a question about that message fills the page with copies
of one paragraph from different replies, and the document that answered it differently — the one
a person most needs to see beside it — is pushed off the end. ``docs/connectors/gmail.md`` §8 has
the measurement.

**The rule.** Walk the ranking best first. A candidate whose document names a conversation is
dropped when at least ``containment`` of its word trigrams already appear in candidates kept
above it from the same conversation, and nothing new sits in its middle: what was not shown lies
only where its chunk began or ended or at a seam between two passages shown apart, never inside a
sentence, where a corrected figure would be. Nothing is reordered and nothing is rescored, so what
survives keeps the rank fusion gave it; the stage only removes what would show a person text
they have already been shown.

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


def trigrams(text: str) -> list[tuple[str, ...]]:
    """The word trigrams of ``text`` in order, with quote markers, case and whitespace folded."""
    words = _WORD.findall(_QUOTE_MARKERS.sub("", text).casefold())
    if len(words) < SHINGLE:
        return [tuple(words)] if words else []
    return [tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """The word trigrams of ``text`` as a set, folded as :func:`trigrams` folds them."""
    return frozenset(trigrams(text))


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
        shown: dict[tuple[str, str], set[tuple[str, ...]]] = {}
        kept: list[Candidate] = []
        for candidate in candidates:
            conversation = conversations.get(candidate.chunk.document_id)
            if conversation is None:
                kept.append(candidate)
                continue
            ordered = trigrams(candidate.chunk.text)
            seen = shown.setdefault(conversation, set())
            if self._repeats(ordered, seen):
                continue
            seen.update(ordered)
            kept.append(candidate)
        return kept

    def _repeats(self, ordered: list[tuple[str, ...]], seen: set[tuple[str, ...]]) -> bool:
        """Whether this text was already shown, give or take its chunk's seams.

        Two conditions. Enough of it appears above — ``containment`` of its distinct trigrams.
        And nothing in its middle is new: one changed or inserted word puts three consecutive
        trigrams that were never shown inside a sentence, so a corrected figure is never dropped
        as a copy of what it corrects. Fewer than three in a row is a seam — the two trigrams
        that straddle the join between two passages shown in different chunks above — and
        unshown text at either end is where this chunk began or ended differently.
        """
        own = set(ordered)
        if not own or len(own & seen) < self._config.containment * len(own):
            return False
        covered = [trigram in seen for trigram in ordered]
        first = covered.index(True)
        last = len(covered) - 1 - covered[::-1].index(True)
        run = 0
        for shown in covered[first : last + 1]:
            run = 0 if shown else run + 1
            if run >= SHINGLE:
                return False
        return True

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


__all__ = ["SHINGLE", "CollapseStage", "shingles", "trigrams"]
