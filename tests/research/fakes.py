"""Fixtures for the research suites, and the components that misbehave on purpose.

Every fake here exists so a guard can be shown to fire. A loop that is only ever driven by a
model returning well-formed JSON certifies nothing about the two things that actually decide
whether this feature is safe: what it does when the plan is nonsense, and what it does when a
generator cannot be given a prompt at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field

from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus
from manicule.core.generation import FinishReason, Token
from manicule.core.retrieval import (
    Candidate,
    Confidence,
    ConfidenceBand,
    Context,
    Filter,
    Query,
)
from manicule.generation.prompt import ChatMessage
from manicule.research.config import ResearchLimits
from manicule.retrieval.retriever import RetrievalResult
from manicule.retrieval.router import Route, Routing

WORKSPACE = "default"


def query(text: str = "how do retries work?") -> Query:
    return Query(text=text, filter=Filter(workspace_ids=frozenset({WORKSPACE})))


def chunk(chunk_id: str, *, document_id: str = "doc-1", text: str = "Retries back off.") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        embed_text=text,
        anchor=HeadingAnchor(path=("Retries",)),
        heading_path=("Retries",),
        kind=BlockKind.PROSE,
        position=0,
        token_count=8,
    )


def candidate(chunk_id: str, *, score: float = 0.5, **kwargs: object) -> Candidate:
    return Candidate(
        chunk=chunk(chunk_id, **kwargs),  # pyright: ignore[reportArgumentType] - keyword plumbing
        score=score,
        scores={"dense": score},
    )


def document(document_id: str = "doc-1") -> Document:
    return Document(
        id=document_id,
        source="filesystem",
        source_id=f"{document_id}-src",
        uri=f"file:///{document_id}",
        title="Retry runbook",
        content_hash=f"hash-{document_id}",
        version_token="v1",  # noqa: S106 - an opaque change token, not a secret
        media_type="text/markdown",
        status=DocumentStatus.INDEXED,
    )


def limits(**overrides: object) -> ResearchLimits:
    """Bounds small enough that a test can reach them."""
    base: dict[str, object] = {"max_cycles": 2, "max_sub_questions": 3, "concurrency": 2}
    base.update(overrides)
    return ResearchLimits(**base)  # pyright: ignore[reportArgumentType] - keyword plumbing


@dataclass
class ScriptedGenerator:
    """Replies with a fixed script, one entry per call, recording what it was asked.

    ``seen`` is the whole point: a loop that sent the wrong prompt and a loop that sent the
    right one produce identical output when the reply is scripted, so the prompt has to be
    observable or the test is asserting nothing about it.
    """

    replies: list[str] = field(default_factory=list[str])
    model_id: str = "fake/model"
    context_window: int = 32768
    seen: list[list[ChatMessage]] = field(default_factory=list[list[ChatMessage]])

    def generate(
        self,
        query: Query,
        context: Context,
        *,
        history: Sequence[ChatMessage] = (),
        documents: Mapping[str, Document] | None = None,
        messages: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[Token]:
        del query, context, history, documents
        self.seen.append(list(messages or ()))
        return self._stream(self.replies.pop(0) if self.replies else '{"queries": []}')

    async def _stream(self, reply: str) -> AsyncIterator[Token]:
        yield Token(text=reply)
        yield Token(finish_reason=FinishReason.STOP)


@dataclass
class PromptlessGenerator:
    """A generator that declares no optional keywords at all.

    A third-party generator is entitled to this — ``messages`` is optional by design — and it
    is exactly the one the loop must refuse rather than quietly hand a citation-protocol answer
    prompt built from a synthetic query.
    """

    model_id: str = "protocol-only/model"
    context_window: int = 32768

    def generate(self, query: Query, context: Context) -> AsyncIterator[Token]:
        del query, context
        return self._stream()

    async def _stream(self) -> AsyncIterator[Token]:
        yield Token(text="{}")
        yield Token(finish_reason=FinishReason.STOP)


@dataclass
class ScriptedRetriever:
    """Returns a fixed ranking per query text, and records the queries it was given.

    ``seen`` is what proves the filter was copied rather than rebuilt: a sub-question that
    reached the store with a different scope is a scope escape, and the only way to see it is
    to keep the query objects.
    """

    rankings: dict[str, list[Candidate]] = field(default_factory=dict[str, list[Candidate]])
    default: list[Candidate] = field(default_factory=list[Candidate])
    seen: list[Query] = field(default_factory=list[Query])
    routed_away: set[str] = field(default_factory=set[str])
    delay_s: float = 0.0
    in_flight: list[int] = field(default_factory=list[int])
    _running: int = 0

    async def retrieve(self, query: Query) -> RetrievalResult:
        self.seen.append(query)
        self._running += 1
        self.in_flight.append(self._running)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            found = self.rankings.get(query.text, self.default)
            routed = query.text in self.routed_away
            return RetrievalResult(
                context=Context(query=query, passages=tuple(found)),
                candidates=list(found),
                confidence=None
                if routed
                else Confidence(
                    score=0.5, band=ConfidenceBand.MEDIUM, reason="a fake, so this is a constant"
                ),
                routing=Routing(route=Route.GREETING if routed else Route.RETRIEVE),
            )
        finally:
            self._running -= 1


__all__ = [
    "WORKSPACE",
    "PromptlessGenerator",
    "ScriptedGenerator",
    "ScriptedRetriever",
    "candidate",
    "chunk",
    "document",
    "limits",
    "query",
]
