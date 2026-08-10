"""The extension points. Everything pluggable in manicule is one of these.

They are :class:`typing.Protocol` definitions, so an implementation satisfies one by having
the right shape — no base class to inherit, no registration decorator, and no import of
manicule from the type's own module if the author would rather not.

None of them requires the lifecycle hooks. ``setup``, ``teardown``, ``health`` and
``metrics`` live in :mod:`manicule.core.lifecycle`, are each detected separately, and are
each optional — a component that needs none of them writes none of them. Folding them in
here would make four methods mandatory on every implementation, which is how a protocol
acquires stubs nobody reads.

This module imports nothing outside the standard library, ``pydantic`` and the sibling
modules of :mod:`manicule.core`. Importing it must never drag in a vector store, a model
runtime or an HTTP client; :mod:`tests.test_import_boundary` fails the build if it does.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from collections.abc import Set as AbstractSet
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from manicule.core.anchors import Anchor
from manicule.core.content import Chunk, Document, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.embedding import EmbedFingerprint, TokenStates, Vector
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.generation import Token
from manicule.core.retrieval import Candidate, Context, Filter, Query
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

# --- ingest ------------------------------------------------------------------------------


@runtime_checkable
class Parser(Protocol):
    """Turns source bytes into located blocks.

    Parsers return *blocks*, not text. Structure — that this is a table, that this is a code
    fence — is visible exactly once, while the markup is still in hand. A parser that
    flattens to a string has destroyed information no downstream component can recover, and
    every attempt to recover it is inference dressed as fact.
    """

    @property
    def media_types(self) -> AbstractSet[str]:
        """IANA media types this parser claims.

        Read-only, so an implementation is free to expose a ``frozenset``, a ``set``, or a
        computed value.
        """
        ...

    def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield blocks in reading order.

        Every block carries an anchor. When the location genuinely cannot be determined,
        that anchor is :class:`~manicule.core.anchors.Unlocated` with a reason — never a
        plausible number.

        Raises:
            ParseError: To decline a document it cannot handle — an encrypted PDF, a
                malformed archive — so the next parser in the fallback chain gets a turn.
                Any other exception fails the document outright.
        """
        ...

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the source text ``anchor`` addresses, or ``None`` if it addresses none.

        This is what makes the round-trip obligation checkable: resolving the anchor on a
        chunk must return the text that chunk claims. A parser that cannot resolve its own
        anchors is emitting locations nobody can verify, which is the failure this whole
        design exists to prevent.

        It takes the retained source bytes — reachable through
        :attr:`~manicule.core.content.Document.original_ref` — rather than a path, so
        verification never means re-fetching.

        ``None`` has one meaning: **there is no text at that location.** It is returned for
        :class:`~manicule.core.anchors.Unlocated`, which addresses nowhere by construction,
        and for a located anchor that does not fit these bytes, which means the anchor and
        the document have diverged. ``None`` rather than ``""`` because an empty string is
        indistinguishable from a genuinely empty region; ``None`` rather than an exception
        because for ``Unlocated`` the answer is expected, not exceptional, and an outcome
        every caller must handle should not arrive as control flow. The second case *is* a
        defect, and :func:`manicule.testing.assert_parser_contract` fails on it.
        """
        ...


@asynccontextmanager
async def parsing(parser: Parser, raw: RawDocument) -> AsyncGenerator[AsyncIterator[ParsedBlock]]:
    """Iterate a parser's blocks, closing the stream on **every** exit path.

    :meth:`Parser.parse` returns an :class:`~collections.abc.AsyncIterator`, and in practice
    every parser implements it as an async *generator*. A generator abandoned part-way — a
    caller that stops early, an assertion that fails between blocks, an exception in the loop
    body — stays suspended, holding whatever it had open at the ``yield``: a document handle,
    a native text page, a decompression stream.

    Nothing collects it promptly. CPython finalises a live async generator through the event
    loop that created it, so one still suspended when that loop closes is finalised late,
    from the wrong loop, after the resources it is about to release have been torn down. The
    observable result is not a warning; it is a crash inside the interpreter's allocator, on
    a stack that names no library anyone here wrote.

    So iteration goes through here, and ``aclose`` runs in a ``finally``::

        async with parsing(parser, raw) as blocks:
            async for block in blocks:
                ...

    Parsers hold their resources in ``with``/``try``-``finally`` around the ``yield`` for the
    same reason: ``aclose()`` throws :class:`GeneratorExit` in at the suspension point, and
    only a ``finally`` runs after that.
    """
    stream = parser.parse(raw)
    try:
        yield stream
    finally:
        await aclose(stream)


async def aclose(stream: AsyncIterator[ParsedBlock]) -> None:
    """Close a block stream if it is a generator, and do nothing if it is not.

    :meth:`Parser.parse` promises an ``AsyncIterator``, which is a weaker thing than an async
    generator and has no ``aclose``. A parser returning a hand-written iterator is a
    legitimate implementation of the protocol, so this asks rather than assumes.
    """
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    await closer()


async def read_blocks(parser: Parser, raw: RawDocument) -> list[ParsedBlock]:
    """Every block a parser produces for a document, with the stream closed afterwards.

    The ordinary way to consume a parser. Use :func:`parsing` instead when the blocks are
    wanted one at a time or the loop may stop early.
    """
    async with parsing(parser, raw) as blocks:
        return [block async for block in blocks]


@runtime_checkable
class Chunker(Protocol):
    """Groups blocks into retrievable chunks.

    Takes the document because a chunk carries its document's identity; taking only blocks
    would push id assignment somewhere that cannot see chunk boundaries.

    **A chunker checks its budget against the embedder.** It resolves the
    :class:`Embedder` as a construction dependency and, in
    :meth:`~manicule.core.lifecycle.Lifecycle.setup`, refuses to start when
    :attr:`fingerprint.max_tokens <manicule.core.fingerprints.ChunkFingerprint.max_tokens>`
    exceeds the embedder's
    :attr:`~manicule.core.embedding.EmbedFingerprint.max_sequence_length`. Past that limit
    the input is truncated silently, and a chunk indexed as its opening tokens still claims
    all of its text — so the check has to happen before ingest, not after.
    """

    fingerprint: ChunkFingerprint
    """Identity of these chunk boundaries. Persisted, so a change can be traced."""

    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        """Produce chunks in document order.

        Implementations respect :attr:`ParsedBlock.kind`: a table or a code block is kept
        whole rather than severed at a character count, and
        :attr:`~manicule.core.content.Chunk.embed_text` carries the heading breadcrumb that
        :attr:`~manicule.core.content.Chunk.text` must not.

        Raises:
            ChunkingError: When the blocks cannot be chunked at all.
        """
        ...


# --- embedding ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. The tier B floor: finished vectors only.

    Tier B backends do their own pooling, so the pooling actually applied cannot be verified
    by inspection — only by measurement. They are admitted on that basis, which is why they
    are the floor rather than the norm.
    """

    fingerprint: EmbedFingerprint
    """Identity of this embedder's output. The sole source of the vector dimension."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed ``texts``, returning one vector per input, in order.

        Every returned vector has length ``fingerprint.dimension``.

        **Callers embedding stored chunks must first call**
        :func:`~manicule.core.embedding.require_within_context`. Text beyond
        ``fingerprint.max_sequence_length`` is dropped by the model with no error raised, so
        an oversized chunk becomes a vector describing its opening while the chunk still
        claims all of its text. The chunker refuses to exceed the limit when it runs — but
        re-embed does not re-chunk, and the fingerprint is unchanged by a limit that fell,
        so that path has no other guard. :func:`manicule.testing.assert_refuses_oversized_chunks`
        holds an implementation to it.
        """
        ...


@runtime_checkable
class TokenStateEmbedder(Embedder, Protocol):
    """Tier A: exposes pre-pooled token states so manicule pools them itself.

    Preferred wherever available. Backends have been observed to bind a
    ``last_hidden_state``-shaped attribute to the already-pooled vector, so a caller
    trusting the field name silently gets the provider's pooling instead of the configured
    one — and CLS versus mean on a typical retrieval model differs by roughly 0.86 cosine
    with no error raised. Reading token states and pooling them here removes that class of
    silent wrongness.
    """

    async def encode(self, texts: Sequence[str]) -> TokenStates:
        """Return per-token hidden states and the attention mask for ``texts``."""
        ...


# --- storage -----------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector storage and nearest-neighbour search."""

    async def ensure_ready(self, fingerprint: EmbedFingerprint) -> None:
        """Prepare the store for vectors from ``fingerprint``.

        The vector table is created here, on first ingest, using the dimension the embedder
        reports — never a constant known ahead of time. On every later call the stored
        fingerprint is compared and a mismatch raises.

        Raises:
            FingerprintMismatchError: When the store already holds vectors from a different
                model. Ingest must not proceed: vectors from two models are not comparable,
                and an index containing both returns confident nonsense.
        """
        ...

    async def fingerprint(self) -> EmbedFingerprint | None:
        """The fingerprint this store was built with, or ``None`` if it holds nothing yet."""
        ...

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Store vectors against chunks, replacing any existing rows for those chunk ids.

        ``chunks`` and ``vectors`` are parallel and must be the same length.
        """
        ...

    async def search(self, vector: Vector, k: int, filter: Filter | None = None) -> list[Candidate]:  # noqa: A002
        """Return up to ``k`` nearest chunks, best first."""
        ...

    async def delete_document(self, document_id: str) -> None:
        """Remove every vector belonging to a document. Idempotent."""
        ...

    async def count(self) -> int:
        """How many vectors are stored."""
        ...


@runtime_checkable
class DocStore(Protocol):
    """Relational storage: documents, chunks, lexical search, sync state.

    Shaped by what ingest and retrieval need. Collections, tags, versions and conversations
    are relational work belonging to the tickets that build those features, and they arrive
    as protocols of their own — one class may implement several. Inventing their signatures
    here would fix shapes before anything needs them.
    """

    async def get_document(self, document_id: str) -> Document | None: ...

    async def find_document(self, source: str, source_id: SourceId) -> Document | None:
        """Look a document up by where it came from, for change detection."""
        ...

    async def upsert_document(self, document: Document) -> Document: ...

    async def set_status(self, document_id: str, status: DocumentStatus, detail: str = "") -> None:
        """Record an outcome. A batch records failures and keeps going."""
        ...

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the vocabulary of the domain
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    async def delete_document(self, document_id: str) -> None:
        """Remove a document and its chunks. Idempotent."""
        ...

    async def replace_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace a document's chunks wholesale. Re-parsing is not additive."""
        ...

    async def get_chunks(self, chunk_ids: Sequence[str]) -> Sequence[Chunk]: ...

    async def search_lexical(
        self,
        text: str,
        k: int,
        filter: Filter | None = None,  # noqa: A002
    ) -> list[Candidate]:
        """BM25 search over stored chunk text. Best first."""
        ...

    async def get_watermark(self, connector: str) -> Watermark | None: ...

    async def set_watermark(self, connector: str, watermark: Watermark) -> None: ...

    def known_source_ids(self, connector: str) -> AsyncIterator[SourceId]:
        """Every source id currently indexed for a connector.

        Reconciliation diffs this against what the source still reports.
        """
        ...


# --- retrieval ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalStage(Protocol):
    """One step of a retrieval pipeline: candidates in, candidates out.

    Every stage has the same signature, so a pipeline is a list of names in configuration
    and two pipelines can be compared without editing code. Dense search, lexical search,
    fusion, filtering and reranking are all this shape — the first stages receive an empty
    list and produce candidates from the query alone.

    A stage may reorder, drop, add, or rescore. It must not mutate the candidates it was
    given; :meth:`~manicule.core.retrieval.Candidate.scored_by` returns a copy.

    .. warning::

       Locked once the evaluation harness exists. Widening the signature later invalidates
       every recorded result, because a stage that receives more can no longer be replayed
       against a run that gave it less.

    Stages needing the query embedded each embed it. The embedding cache is keyed by model
    identity and text, so the repeat is free — and the alternative, threading shared state
    between stages, makes a pipeline order-dependent and therefore not comparable.
    """

    name: str
    """Stable identifier. Names the stage in configuration and keys its score history."""

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]: ...


@runtime_checkable
class Reranker(RetrievalStage, Protocol):
    """A retrieval stage that rescores candidates with a dedicated model.

    Structurally a stage, so it drops into a pipeline like any other. The extra attribute
    exists so a recorded evaluation result says which model produced the ranking; a result
    that cannot name its reranker cannot be reproduced.
    """

    model_id: str


# --- generation --------------------------------------------------------------------------


@runtime_checkable
class Generator(Protocol):
    """Produces an answer from a query and assembled context.

    One interface for local and hosted models alike, selected by ``base_url``. There is no
    per-provider variant, here or anywhere.
    """

    model_id: str

    def generate(self, query: Query, context: Context) -> AsyncIterator[Token]:
        """Stream the answer. The final token carries a finish reason."""
        ...


# --- connectors --------------------------------------------------------------------------


@runtime_checkable
class Connector(Protocol):
    """Finds documents in a source and fetches them."""

    name: str

    def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield documents created or changed since ``watermark``.

        ``None`` means no previous sync: yield everything. Implementations use the source's
        own change signal rather than re-enumerating and hashing, which is the difference
        between a sync that costs what changed and one that costs the whole corpus.
        """
        ...

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Retrieve one document's bytes."""
        ...

    def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield every source id that still exists.

        Part of the protocol, not an optional extra. Incremental sync cannot detect
        deletions — a deleted page simply stops appearing — so without this pass the index
        serves removed documents indefinitely, and no amount of syncing fixes it. Making it
        mandatory means no connector can quietly omit it.
        """
        ...


# --- middleware --------------------------------------------------------------------------


@runtime_checkable
class Middleware(Protocol):
    """Observes and transforms documents as they pass through ingest.

    Every hook is **transformational**: it receives a value and returns the value the
    pipeline continues with. There is no in-place mutation contract and no discarded return
    — a hook that returns a new object has an effect, which is the only behaviour anybody
    expects from it.

    ``before_parse`` may return ``None`` to drop a document, which records it as
    :attr:`~manicule.core.content.DocumentStatus.SKIPPED`. That is the only short-circuit;
    a hook that wants to abort for any other reason raises.

    **``Chunk.text`` is immutable after parse. ``Chunk.embed_text`` is not.**

    This is the one hard limit on what a hook may do, and it exists because no parser test
    can catch a violation of it. Every parser is held to a round-trip obligation — resolving
    a chunk's anchor returns the text the chunk claims (:func:`assert_parser_contract
    <manicule.testing.assert_parser_contract>`). A middleware that rewrites ``text`` breaks
    that correspondence *after every one of those tests has passed*, leaving a corpus that is
    internally consistent while its citations quote text the source document does not
    contain.

    ``embed_text`` is different in kind: never cited, never displayed, and shaped for
    retrieval rather than for reproduction. Rewriting it is legitimate — redaction and
    context augmentation both belong there — and it changes every vector, which is why a
    middleware that does so declares :attr:`mutates_embedded_text` and is folded into the
    chunk fingerprint.

    So: rewrite ``embed_text`` freely and declare it; return ``text`` unchanged.
    :func:`assert_middleware_contract <manicule.testing.assert_middleware_contract>` enforces
    this, because an unenforced guarantee is worse than an absent one.

    Hooks run in the order middleware is listed in configuration. Order is declared where a
    reader can see it, rather than emerging from priority numbers spread across packages.

    Inherit from this to pick up the pass-through defaults and override only the hooks you
    care about.
    """

    name: str

    mutates_embedded_text: bool = False
    """Whether any hook rewrites ``Chunk.embed_text``.

    Declared rather than detected, because detection would only catch the mutations that
    happened to fire on the documents someone tested. The ingest pipeline folds the sorted
    ``(name, version)`` set of middleware declaring this into the chunk fingerprint it
    compares at startup — otherwise a middleware that rewrites embedded text changes every
    vector while both fingerprint refusals pass, since neither knows middleware exists.

    Leaving this ``False`` while mutating ``embed_text`` does not corrupt a citation, but it
    does make a corpus that no fingerprint describes.
    """

    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        """Transform, or return ``None`` to skip the document."""
        return raw

    async def after_parse(self, document: Document, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        return blocks

    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Last chance to alter what gets embedded and stored.

        Redaction belongs here or nowhere: this is the point where "what leaves the machine"
        and "what is written to the index" are still the same list.
        """
        return chunks

    async def after_store(self, document: Document) -> None:
        """Observe a completed document. The return value is deliberately nothing."""
        return


__all__ = [
    "Chunker",
    "Connector",
    "DocStore",
    "Embedder",
    "Generator",
    "Middleware",
    "Parser",
    "Reranker",
    "RetrievalStage",
    "TokenStateEmbedder",
    "VectorStore",
    "aclose",
    "parsing",
    "read_blocks",
]
