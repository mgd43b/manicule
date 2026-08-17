"""Fixtures and deliberately-misbehaving components for the generation suites.

Every fake here exists so a guard can be shown to fire. A citation suite in which every
citation is valid certifies nothing: it passes identically against a verifier that returns
``True`` unconditionally.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from manicule.config.settings import Settings
from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Chunk, Document, DocumentStatus, RawDocument
from manicule.core.generation import FinishReason, Token, Usage
from manicule.core.retrieval import Candidate, Context, Filter, Query
from manicule.generation.prompt import ChatMessage
from manicule.generation.verification import ChainRouter, OpenSource, RetainedBytesResolver

WORKSPACE = "default"


def query(text: str = "how do we roll back a deploy?") -> Query:
    return Query(text=text, filter=Filter(workspace_ids=frozenset({WORKSPACE})))


def chunk(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    text: str = "Roll back with `deploy --rollback`.",
    anchor: Anchor | None = None,
    heading_path: tuple[str, ...] = ("Operations", "Rollback"),
    position: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        embed_text=" > ".join((*heading_path, text)),
        anchor=anchor or HeadingAnchor(path=heading_path),
        heading_path=heading_path,
        kind=BlockKind.PROSE,
        position=position,
        token_count=16,
    )


def candidate(**kwargs: Any) -> Candidate:
    return Candidate(chunk=chunk(**kwargs), score=0.9)


def document(
    *,
    document_id: str = "doc-1",
    source: str = "confluence",
    title: str = "Deploy runbook",
    original_ref: str | None = "blob-1",
    media_type: str = "text/markdown",
) -> Document:
    return Document(
        id=document_id,
        source=source,
        source_id=f"{document_id}-src",
        uri=f"https://example.invalid/{document_id}",
        title=title,
        content_hash=f"hash-{document_id}",
        version_token="v1",  # noqa: S106 - a connector's opaque change token, not a secret
        original_ref=original_ref,
        media_type=media_type,
        status=DocumentStatus.INDEXED,
        metadata={"parser_used": "markdown"},
    )


def context(passages: Sequence[Candidate] | None = None, **kwargs: Any) -> Context:
    return Context(
        query=query(),
        passages=tuple(passages if passages is not None else (candidate(),)),
        token_count=64,
        **kwargs,
    )


@dataclass(slots=True)
class FakeDocuments:
    """A document lookup over a fixed map."""

    documents: dict[str, Document] = field(default_factory=dict[str, Document])

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)


@dataclass(slots=True)
class FakeBlobs:
    """A blob store. A missing digest is a real outcome, not an error."""

    blobs: dict[str, bytes] = field(default_factory=dict[str, bytes])

    async def get(self, digest: str) -> bytes | None:
        return self.blobs.get(digest)


@dataclass(slots=True)
class FakeParser:
    """Resolves a heading anchor by looking it up in a table.

    ``resolutions`` maps a heading path's last element to the text at that location, so a
    fixture can make an anchor resolve to the wrong text — the drift this whole ladder exists
    to catch — without needing a real document format.
    """

    resolutions: dict[str, str | None] = field(default_factory=dict[str, str | None])
    media_types: frozenset[str] = frozenset({"text/markdown"})
    calls: list[Anchor] = field(default_factory=list[Anchor])

    def parse(self, raw: RawDocument) -> AsyncIterator[Any]:  # pragma: no cover - never called
        raise NotImplementedError

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del raw
        self.calls.append(anchor)
        if isinstance(anchor, Unlocated):
            return None
        if isinstance(anchor, HeadingAnchor):
            return self.resolutions.get(anchor.path[-1])
        return None


@dataclass(slots=True)
class FakeChain:
    """The part of a parser chain the router reads."""

    parser: FakeParser

    @property
    def parsers(self) -> Mapping[str, Any]:
        return {"markdown": self.parser}

    def resolve(self, media_type: str) -> tuple[str, ...]:
        del media_type
        return ("markdown",)


def resolver(parser: FakeParser, blobs: FakeBlobs | None = None) -> RetainedBytesResolver:
    store = blobs or FakeBlobs({"blob-1": b"# Rollback\nRoll back with `deploy --rollback`.\n"})
    return RetainedBytesResolver(blobs=store, router=ChainRouter(FakeChain(parser)))


@dataclass(slots=True)
class BrokenResolver:
    """A resolver whose parse raises. A parser bug must drop citations, never answers."""

    can_resolve: bool = True

    async def open(self, document: Document) -> OpenSource | None:
        del document
        msg = "this parser is broken"
        raise RuntimeError(msg)


@dataclass(slots=True)
class SlowResolver:
    """A resolver that never returns, for exercising the verification deadline."""

    can_resolve: bool = True

    async def open(self, document: Document) -> OpenSource | None:
        del document
        await asyncio.Event().wait()
        raise AssertionError  # pragma: no cover - unreachable


@dataclass(slots=True)
class ScriptedGenerator:
    """A generator that emits a fixed script, with optional history support.

    ``supports_history`` decides whether ``generate`` declares the optional keyword, which is
    what the answer path inspects — so a generator that cannot carry a conversation is
    exercised as a first-class case rather than assumed away.
    """

    script: Sequence[str] = ()
    model_id: str = "fake/model"
    context_window: int = 32768
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage | None = None
    fail_after: int | None = None
    supports_history: bool = True
    seen_history: list[ChatMessage] = field(default_factory=list[ChatMessage])
    seen_context: list[Context] = field(default_factory=list[Context])
    seen_query: list[Query] = field(default_factory=list[Query])
    seen_messages: list[ChatMessage] = field(default_factory=list[ChatMessage])
    seen_documents: list[Mapping[str, Document]] = field(
        default_factory=list[Mapping[str, Document]]
    )
    closed: bool = False

    def generate(
        self,
        query: Query,
        context: Context,
        *,
        history: Sequence[ChatMessage] = (),
        documents: Mapping[str, Document] | None = None,
        messages: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[Token]:
        """Records the prompt as well as the parts.

        The earlier fake did ``del documents`` and never saw ``messages`` at all, which is why
        a redaction test could pass while the labels reaching the provider were unredacted:
        the fake could not observe the channel the defect was on.
        """
        self.seen_documents.append(dict(documents or {}))
        self.seen_messages.extend(messages or ())
        if not self.supports_history:
            history = ()
        self.seen_history.extend(history)
        self.seen_context.append(context)
        self.seen_query.append(query)
        return self._stream()

    async def _stream(self) -> AsyncIterator[Token]:
        try:
            for index, piece in enumerate(self.script):
                if self.fail_after is not None and index == self.fail_after:
                    msg = "the provider stopped half-way"
                    raise RuntimeError(msg)
                yield Token(text=piece)
            yield Token(finish_reason=self.finish_reason, usage=self.usage)
        finally:
            self.closed = True


@dataclass(slots=True)
class ProtocolOnlyGenerator:
    """A generator written strictly to the protocol: no optional keywords at all."""

    model_id: str = "plugin/model"
    context_window: int = 8192
    script: Sequence[str] = ()

    def generate(self, query: Query, context: Context) -> AsyncIterator[Token]:
        del query, context
        return self._stream()

    async def _stream(self) -> AsyncIterator[Token]:
        for piece in self.script:
            yield Token(text=piece)
        yield Token(finish_reason=FinishReason.STOP)


def settings(**overrides: Any) -> Settings:
    """Settings with nothing reaching the network unless a test asks for it."""
    base: dict[str, Any] = {
        "llm": {"provider": "ollama", "model": "qwen2.5:14b"},
        "embedding": {"provider": "onnx"},
    }
    base.update(overrides)
    return Settings(**base)


__all__ = [
    "WORKSPACE",
    "BrokenResolver",
    "FakeBlobs",
    "FakeChain",
    "FakeDocuments",
    "FakeParser",
    "ProtocolOnlyGenerator",
    "ScriptedGenerator",
    "SlowResolver",
    "candidate",
    "chunk",
    "context",
    "document",
    "query",
    "resolver",
    "settings",
]
