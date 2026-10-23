"""Near-duplicate collapse within a conversation: a passage a thread repeats is shown once.

A mail message keeps the reply chain it quotes, so a thread of thirty messages carries its first
message thirty times. Ranked per chunk, a question about that message fills the page with copies
of one paragraph from different replies, and the document that answered it differently — the one
a person most needs to see beside it — is pushed off the end. ``docs/connectors/gmail.md`` §8 has
the measurement.

**The rule.** Walk the ranking best first. A candidate whose document names a conversation is
dropped when at least ``containment`` of its word trigrams already appear in candidates kept
above it from the same conversation. Nothing is reordered and nothing is rescored, so what
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

if TYPE_CHECKING:
    from manicule.core.content import Metadata
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Candidate, Query

SHINGLE = 3
"""Words per shingle. Long enough that two passages sharing a phrase are not the same passage,
short enough that one wrapped differently still is."""

_QUOTE_MARKERS = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.MULTILINE)
_WORD = re.compile(r"\w+")


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """The word trigrams of ``text``, with quote markers, case and whitespace folded away."""
    words = _WORD.findall(_QUOTE_MARKERS.sub("", text).casefold())
    if len(words) < SHINGLE:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1))


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
        shown: dict[str, set[tuple[str, ...]]] = {}
        kept: list[Candidate] = []
        for candidate in candidates:
            conversation = conversations.get(candidate.chunk.document_id)
            if conversation is None:
                kept.append(candidate)
                continue
            own = shingles(candidate.chunk.text)
            seen = shown.setdefault(conversation, set())
            if own and len(own & seen) >= self._config.containment * len(own):
                continue
            seen |= own
            kept.append(candidate)
        return kept

    async def _conversations(self, query: Query, candidates: list[Candidate]) -> dict[str, str]:
        """The conversation each candidate's document belongs to, where it names one.

        Read under the query's own filter, narrowed to these documents, so this is no wider a
        read than the one that produced the candidates.
        """
        wanted = frozenset(candidate.chunk.document_id for candidate in candidates)
        if query.filter.document_ids:
            wanted &= query.filter.document_ids
        if not wanted:
            return {}
        documents = await self._docstore.list_documents(
            query.filter.model_copy(update={"document_ids": wanted}), limit=len(wanted)
        )
        key = self._config.thread_key
        return {
            document.id: str(document.metadata[key])
            for document in documents
            if document.metadata.get(key) not in (None, "")
        }


__all__ = ["SHINGLE", "CollapseStage", "shingles"]
