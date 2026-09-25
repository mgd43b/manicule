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

import asyncio
import inspect
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Iterable,
    Mapping,
    Sequence,
)
from collections.abc import Set as AbstractSet
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from manicule.core.anchors import Anchor
from manicule.core.ann import AnnIndexBuild, AnnIndexState
from manicule.core.content import Chunk, Document, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.embedding import (
    EmbedFingerprint,
    StoredVector,
    TokenStates,
    Vector,
    VectorChecksumBackfill,
    VectorChecksumCoverage,
)
from manicule.core.fingerprints import ChunkFingerprint, RelationRules
from manicule.core.generation import Token
from manicule.core.organization import (
    ChunkEdge,
    ChunkRelationType,
    CitationResolution,
    Collection,
    CollectionRule,
    DocumentVersion,
    Restoration,
    Tag,
    TrashEntry,
)
from manicule.core.retrieval import Candidate, Context, Filter, Query
from manicule.core.sources import (
    DiscoveredDoc,
    DocRef,
    EnumerationProgress,
    SourceId,
    Watermark,
)

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


_ABANDONED: set[asyncio.Future[None]] = set()
"""Closes that outlived their deadline.

A reference is held so the task is not collected mid-flight, and its outcome is consumed so a
failure does not surface later as an unhandled-task warning from a stack naming nobody.
"""


async def bounded(awaitable: Awaitable[None], deadline_s: float | None) -> None:
    """Await ``awaitable``, **abandoning** it past ``timeout`` rather than waiting for it.

    Deliberately not :func:`asyncio.wait_for`, which on expiry cancels the inner awaitable and
    then waits for that cancellation to finish — so an awaitable that catches
    :exc:`asyncio.CancelledError`, as a retry loop or a driver teardown does, blocks past the
    deadline anyway. The same fact the parse workers had to establish: ``wait_for`` cancels the
    await, not the work.

    Past the deadline the task is canceled and left. Whatever it holds goes to its own pool's
    teardown, which is the trade this bound was always making: a leaked socket is recovered by
    the pool, a shutdown that never returns is recovered by nothing.

    Exceptions are swallowed. Every caller is a ``finally``, frequently one already unwinding a
    cancellation, and raising would replace the failure somebody needs to see with the failure
    of the tidy-up after it.
    """
    running = asyncio.ensure_future(awaitable)
    try:
        await asyncio.wait({running}, timeout=deadline_s)
    except asyncio.CancelledError:
        running.cancel()
        _abandon(running)
        raise
    if not running.done():
        running.cancel()
        _abandon(running)
        return
    if not running.cancelled():
        running.exception()


def _abandon(task: asyncio.Future[None]) -> None:
    _ABANDONED.add(task)
    task.add_done_callback(_ABANDONED.discard)
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())


@asynccontextmanager
async def parsing(parser: Parser, raw: RawDocument) -> AsyncGenerator[AsyncIterator[ParsedBlock]]:
    """Iterate a parser's blocks, closing the stream on **every** exit path.

    :meth:`Parser.parse` returns an :class:`~collections.abc.AsyncIterator`, and in practice
    every parser implements it as an async *generator*. A generator abandoned part-way — a
    caller that stops early, an assertion that fails between blocks, an exception in the loop
    body — stays suspended, holding whatever it had open at the ``yield``: a document handle,
    a native text page, a decompression stream.

    Nothing collects it promptly. CPython finalizes a live async generator through the event
    loop that created it, so one still suspended when that loop closes is finalized late,
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


async def aclose[T](stream: AsyncIterator[T], *, timeout: float | None = None) -> None:  # noqa: ASYNC109 - the deadline is enforced here with wait_for; there is nothing below to pass it to
    """Close a stream if it is a generator, and do nothing if it is not.

    :meth:`Parser.parse` and :meth:`Generator.generate` both promise an ``AsyncIterator``,
    which is a weaker thing than an async generator and has no ``aclose``. An implementation
    returning a hand-written iterator satisfies either protocol, so this asks rather than
    assumes.

    Args:
        stream: The iterator to close.
        timeout: A hard deadline on the close, or ``None`` for none. A parser closes local
            handles and needs none; a generator holds a provider connection, and a shutdown
            path that can block indefinitely on a misbehaving remote server is a worse failure
            than a leaked socket. Past the deadline the close is abandoned to the connection
            pool's own teardown and this returns.
    """
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    if timeout is None:
        await closer()
        return
    # **Not `wait_for`.** On expiry `wait_for` cancels the inner awaitable and then *waits for
    # that cancellation to finish*, so a closer that catches `CancelledError` — a retry loop, a
    # driver that swallows it during teardown — blocks past the deadline anyway, which is
    # exactly the shutdown this bound exists to guarantee. The same fact the parse workers had
    # to establish: `wait_for` cancels the await, not the work.
    #
    # So the close runs as a task, and past the deadline the task is canceled and **left**.
    # The connection goes to the pool's own teardown, which is the trade this bound was always
    # making: a leaked socket is recovered by the pool, a shutdown that never returns is
    # recovered by nothing.
    await bounded(closer(), timeout)


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
    """Dense vector storage and nearest-neighbor search."""

    async def ensure_ready(
        self, fingerprint: EmbedFingerprint, *, embed_text_middleware: Sequence[str] = ()
    ) -> None:
        """Prepare the store for vectors from ``fingerprint``.

        The vector table is created here, on first ingest, using the dimension the embedder
        reports — never a constant known ahead of time. On every later call the stored
        fingerprint is compared and a mismatch raises.

        ``embed_text_middleware`` is
        :attr:`~manicule.core.fingerprints.ChunkFingerprint.embed_text_middleware`: the sorted
        ``name@version`` of every middleware that declares it rewrites embedded text. The store
        needs it because it, and not its caller, derives the embedding-input identity it stores
        beside each vector (:func:`~manicule.core.embedding.embedding_input_identity`). The
        default is correct for a configuration in which no middleware declares the capability,
        which is every configuration that has not opted in.

        Raises:
            FingerprintMismatchError: When the store already holds vectors from a different
                model. Ingest must not proceed: vectors from two models are not comparable,
                and an index containing both returns confident nonsense.
        """
        ...

    async def fingerprint(self) -> EmbedFingerprint | None:
        """The fingerprint this store was built with, or ``None`` if it holds nothing yet."""
        ...

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        publication_id: str = "legacy",
    ) -> None:
        """Store vectors against chunks in one publication generation.

        ``chunks`` and ``vectors`` are parallel and must be the same length.
        A physical vector row is keyed by ``publication_id`` plus logical chunk id, so staging
        a replacement cannot overwrite the generation retrieval still serves.

        The store records each vector's embedding-input identity alongside it, derived from
        the chunk's ``embed_text``, the fingerprint the store was prepared with, and the
        middleware declaration it was given. Derived here rather than supplied by the caller
        so that there is one rule for writing it and one rule for reading it back —
        :meth:`stored_vectors` is the read, and two places computing one identity is how the
        two come to disagree.
        """
        ...

    async def stored_vectors(self, chunks: Sequence[Chunk]) -> Mapping[str, StoredVector]:
        """What this store holds for each of ``chunks``, and whether it can still be used.

        Asked about *chunks* rather than chunk ids, deliberately. A chunk id is derived from
        ``text`` while a vector is produced from ``embed_text``, so a store handed only an id
        could answer nothing better than "a row exists" — and reuse on that answer preserves a
        stale vector under current text, which is the one failure this whole path exists to
        avoid. Handing over the chunk is what lets the store compare the embedding input.

        Returns:
            One :class:`~manicule.core.embedding.StoredVector` per chunk, keyed by chunk id,
            with an entry for **every** chunk asked about — including the ones with no row, so
            a caller never has to decide what a missing key meant. A vector is returned only
            with :attr:`~manicule.core.embedding.VectorState.READABLE`, and is the stored
            vector exactly as stored.
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
class AnnIndexMaintenance(Protocol):
    """Optionally reports and maintains an approximate-nearest-neighbor index.

    Optional the same way :class:`EnumerationProgressConnector` is, and for the same reason:
    the capability belongs to a backend that has one, not to every backend. Exhaustive search
    is exact, so a store that never builds an index is not a degraded store — it is a store
    whose search is always the reference answer, and requiring it to implement two methods
    that can only refuse would make the protocol describe a lifecycle it does not have.

    A store without this capability reports no index state at all rather than an empty one.
    ``docs/storage.md`` §6.2 is the lifecycle these two methods execute.
    """

    async def ann_index_state(self, *, threshold: int) -> AnnIndexState:
        """Whether search is exhaustive, indexed, or overdue — measured, never recorded."""
        ...

    async def build_ann_index(
        self, *, threshold: int, force: bool = False, dry_run: bool = False
    ) -> AnnIndexBuild:
        """Bring the index up to what ``threshold`` asks for, without interrupting search."""
        ...


@runtime_checkable
class VectorIntegrityMaintenance(Protocol):
    """Optionally reports and repairs the numerical-integrity coverage of stored vectors.

    Optional on the same terms as :class:`AnnIndexMaintenance`: the capability belongs to a
    backend that persists vectors somewhere a checksum can be written beside them, not to every
    backend. A store without it reports no coverage at all rather than a coverage of nothing —
    ``docs/storage.md`` §6.2.5 is the contract these two methods execute.

    Neither method may call an embedder, a parser, a connector or a source system. That is the
    property the whole design rests on: numerical integrity is established from bytes already on
    disk, which is why it can be checked routinely and re-embedding cannot.
    """

    async def checksum_coverage(
        self, *, recompute: bool = False, page_size: int = 512
    ) -> VectorChecksumCoverage:
        """How many stored vectors carry a checksum, and — with ``recompute`` — still match.

        Counting is cheap enough for a status page; recomputing is a bounded scan of the corpus
        and is what an operator asks for. Aggregate counts only: no checksum value, no vector
        component and no chunk identifier crosses this boundary.
        """
        ...

    async def backfill_checksums(
        self, *, limit: int = 512, dry_run: bool = False
    ) -> VectorChecksumBackfill:
        """Give one bounded page of pre-checksum rows the checksum their stored vector implies.

        Resumable and idempotent: the page is selected by "records no checksum", so a finished
        row is not a row the next pass can see and there is no cursor to lose.
        """
        ...


@runtime_checkable
class PublicationAwareVectorStore(Protocol):
    """Optionally groups its rows into publications it can count, validate, copy and retire.

    Optional on the same terms as :class:`AnnIndexMaintenance`: the capability belongs to a
    backend that holds several physical publications of one logical chunk, not to every
    backend. A store without it is not a degraded store — it holds exactly one row per chunk,
    which is why it has nothing to name. ``docs/storage.md`` §6.5 is the lifecycle these
    methods serve.

    **This protocol exists so that asking the question does not cost a native library.** The
    publication surface is LanceDB's, and :mod:`manicule.storage.vectors` imports ``lancedb``
    at module scope — so a backend-agnostic path that narrowed with
    ``isinstance(store, LanceVectorStore)`` had to dlopen LanceDB's extension to find out that
    the answer was no. On a CPU without AVX2 that is not a slow import but a ``SIGILL``, which
    made the Qdrant backend unusable on the hardware it exists to serve. A capability is asked
    of the object, never of the module that would implement it.
    """

    async def publication_row_count(self, publication_id: str) -> int:
        """How many rows one publication holds, for validation and for cleanup accounting."""
        ...

    async def delete_publication(self, publication_id: str) -> int:
        """Retire one publication's rows, reporting how many there were."""
        ...

    async def prepare_publication_validation(self, publication_id: str) -> None:
        """Make whatever a completeness check needs selective enough to run repeatedly."""
        ...

    async def publication_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """Whether every one of ``chunks`` is present in ``publication_id``'s space."""
        ...

    async def publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        """The same question over one page, so a corpus of any size stays constant-memory."""
        ...

    async def copy_publication(
        self,
        source_publication_id: str,
        target_publication_id: str,
        chunks: Sequence[Chunk],
    ) -> None:
        """Carry ``chunks`` into a second publication without re-embedding them."""
        ...


@runtime_checkable
class PublicationBoundVectorStore(PublicationAwareVectorStore, Protocol):
    """A publication-aware handle that follows the durable pointer between operations.

    The narrower capability, and the one the application runtime hands out: it resolves which
    physical generation is live by reading the pointer the document store owns, so a caller
    holds a handle to "this workspace's vectors" rather than to a directory. Extends
    :class:`PublicationAwareVectorStore` the way :class:`TokenStateEmbedder` extends
    :class:`Embedder` — the same subject, asked for more.

    The ``vector_table`` argument each deletion takes is what makes it *bound*: cleanup names
    the generation it believes it is deleting from, and a handle that has since followed the
    pointer elsewhere refuses rather than deleting the wrong generation's rows.

    Durable re-embedding's shadow generations are this capability, which is why a re-embed on a
    backend without it is refused by name rather than half-performed — ``docs/storage.md``
    §6.5, and §6.7 for the backend that does not have it.
    """

    async def teardown(self) -> None:
        """Release the handle's open resources without disturbing what is on disk."""
        ...

    async def delete_bound_chunks(self, vector_table: str | None, vector_ids: Sequence[str]) -> int:
        """Delete rows by id from the named generation, refusing if this handle has moved."""
        ...

    async def delete_bound_publication(self, vector_table: str | None, publication_id: str) -> int:
        """Retire a whole publication from the named generation, under the same refusal."""
        ...


@runtime_checkable
class ResettableVectorStore(Protocol):
    """Optionally discards everything it holds for one workspace, recorded identity included.

    Optional on the same terms as :class:`AnnIndexMaintenance`, and asked of the object for the
    same reason the rest are. A derived reset deletes the rows the relational store tombstoned,
    which needs no capability at all; what a tombstone cannot name is the storage *around* those
    rows — a collection, its payload indexes, and the record saying which model this workspace's
    vectors came from. Leaving that behind is not a cosmetic residue: the next ingest under a
    different embedder is refused by a fingerprint the reset was supposed to have removed, which
    is the state ``reset-index`` exists to get an installation out of.

    The embedded backend does not implement this and is not degraded by it. Its storage is a
    directory the runtime itself owns, and removing it depends on something the store does not
    know — whether other workspaces still share the upgrade-era root (``docs/storage.md`` §6.5).
    A networked backend is the opposite case and the one this exists for: manicule holds a
    client rather than a filesystem, so the only thing that can drop a collection is the store.

    A backend without this capability still resets. The tombstone sweep is backend-agnostic, and
    the reset reports that no physical storage was removed rather than implying that it was.
    """

    async def reset_storage(self) -> bool:
        """Discard this workspace's vectors and the identity recorded beside them. Idempotent.

        Scoped to one workspace, never to the server: an installation may share a Qdrant
        instance or a directory with corpora this reset was not asked about, and a reset that
        took them with it would be the more expensive version of the problem it solves.

        Returns:
            Whether there was anything to remove, so a reset reports a physical removal it
            actually performed. A workspace that never held a vector answers ``False``.
        """
        ...


@runtime_checkable
class MultiWorkspaceVectorStore(Protocol):
    """Optionally opens a read handle on another workspace's vectors, held by the same backend.

    What an administrator's cross-workspace search needs from a backend that is not a directory
    (``docs/retrieval.md`` §3.2): one scoped handle per workspace, fanned out and merged. The
    embedded backend does not implement this and is not degraded by it — each workspace is its
    own directory, and the application runtime opens that directory itself, exactly as it opens
    the serving workspace's. A networked backend holds one client for every workspace, so the
    handle for a second workspace is a second view over the same connection rather than a second
    connection, and only the store knows how to make one.

    **It opens; it never prepares.** :meth:`VectorStore.ensure_ready` is a write the first time a
    workspace is prepared — it creates a collection and records which model filled it — and a
    search that spans workspaces is a read of corpora the caller is not ingesting into. So a
    workspace that holds nothing is refused rather than created, and one whose recorded model is
    not ``fingerprint`` is refused rather than searched: two workspaces' cosines are comparable
    only when one model produced both, which is the whole reason the merge rule may use them.
    """

    async def open_workspace(self, workspace_id: str, fingerprint: EmbedFingerprint) -> VectorStore:
        """A read handle on ``workspace_id``'s vectors, ready to search in ``fingerprint``'s space.

        The handle shares this store's connection and does not own it: tearing it down leaves
        the connection to whoever opened it.

        Raises:
            VectorStoreStateError: The workspace holds no vectors in this backend.
            FingerprintMismatchError: Its vectors came from a different model, including a
                different model of the same size.
        """
        ...


@runtime_checkable
class InspectableVectorStore(VectorStore, Protocol):
    """Optionally streams every row it holds, in the shape ``vector_schema`` names.

    The read half of moving a corpus between backends (``docs/storage.md`` §6.8). What makes
    that a copy rather than a conversion is that there is one row shape and
    :mod:`manicule.storage.vector_schema` is where it is written down — so a store that can
    hand its rows back in that shape can be read by a migration that has never heard of it.

    Optional on the same terms as :class:`AnnIndexMaintenance`: a backend that can only be
    queried by vector, and never enumerated, is not a degraded store. It is one whose contents
    can be rebuilt but not carried, which is a real difference to tell an operator about before
    they plan a move rather than after.
    """

    def inspection_pages(self, *, page_size: int = 256) -> AsyncIterator[list[dict[str, Any]]]:
        """Every row this store holds, in bounded batches, in a stable order.

        Stable because a migration reports a count against what it read: an order that varied
        between the count and the read would make a short copy indistinguishable from a
        concurrent write. Bounded because the corpora worth moving do not fit in memory.
        """
        ...


@runtime_checkable
class AdoptingVectorStore(VectorStore, Protocol):
    """Optionally takes rows another backend already holds, without re-embedding them.

    The write half of :class:`InspectableVectorStore`, and the pair is deliberately two
    protocols rather than one: the backends are not symmetric. The embedded store is the one
    installations start on and therefore the one they migrate *from*; a networked store is the
    one they migrate *to*. A single ``MigratableVectorStore`` would make every backend claim
    both halves in order to offer either.

    **Why a store must implement this rather than a migration calling ``upsert``.** ``upsert``
    takes a :class:`~manicule.core.content.Chunk` and a vector fresh from an embedder, and
    derives the row's checksum and embedding identity on the way past. Both derivations are
    right there and wrong here: a checksum recomputed over a vector that arrived from another
    store would describe whatever arrived, so a source row whose numbers had drifted would be
    written with a fresh digest certifying the drift. Adoption carries the recorded pair
    instead, which is what leaves the integrity check still able to notice.
    """

    def storage_name(self, fingerprint: EmbedFingerprint) -> str:
        """What this backend calls the place ``fingerprint``'s vectors live.

        A Lance table, a Qdrant collection, a relational table — the physical name in this
        backend's own namespace, which is not :func:`~manicule.storage.vector_schema.space_name`
        because a backend may qualify that with a prefix or a schema. An operation that moved a
        corpus and then could not say where it went would leave the operator reconstructing the
        name from a configuration file.
        """
        ...

    async def rows_in_space(self, fingerprint: EmbedFingerprint) -> int:
        """How many rows this backend holds for ``fingerprint``'s space, right now.

        **Asked about the space rather than about the store, and that distinction is the whole
        reason this exists.** :meth:`VectorStore.count` answers for whatever space the backend
        has *recorded* itself as holding, so a destination whose recorded identity is missing —
        a half-finished reset, a metadata record removed by hand — answers zero while its
        storage is still full. A migration's refusal to merge into a populated destination
        would then be satisfied by exactly the case it exists to catch, and rows of unknown
        provenance would be mixed into the corpus being moved.

        Zero for a space this backend has never created, which is the ordinary answer on a
        first migration and is not an error.
        """
        ...

    async def adopt_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Store rows keyed by the ``vector_schema`` names, carrying every field across.

        Returns:
            How many rows were written.
        """
        ...


@runtime_checkable
class DocStore(Protocol):
    """Relational storage: documents, chunks, lexical search, sync state.

    Shaped by what ingest and retrieval need. Organization on top of the corpus arrives as
    protocols of its own — :class:`CollectionStore`, :class:`TagStore`, :class:`VersionStore`,
    :class:`TrashStore` and :class:`ChunkRelationStore` — and **one class may implement
    several**; :class:`manicule.storage.docstore.SqliteDocStore` implements all six. Splitting
    them keeps a component's dependency honest: a retrieval stage that needs to resolve a
    collection into document ids asks for a :class:`CollectionStore` and cannot reach a
    document's chunks with the handle it was given.

    Conversations are the one member of that list still absent, and belong to the ticket that
    builds chat.
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


# --- organization --------------------------------------------------------------------------
#
# Five protocols rather than five more methods on DocStore, and the split is not cosmetic.
# Every one of them is workspace-scoped in exactly the way DocStore is: the handle carries the
# workspace, no method takes one, and anything naming a document from another tenant is
# refused rather than ignored. A store that silently skipped a foreign document id would make
# "add these forty documents to a collection" report success having added thirty-nine.


@runtime_checkable
class CollectionStore(Protocol):
    """Named sets of documents: manual membership, rule-driven membership, or both.

    **Membership is evaluated, not materialized.** A collection with a
    :class:`~manicule.core.organization.CollectionRule` reports the documents the rule selects
    *now*, unioned with whatever was added by hand. Materializing the rule's answer at write
    time would make the collection a snapshot with a name that promises otherwise, and there
    would be nothing to notice it had gone stale.
    """

    async def create_collection(
        self, name: str, *, description: str | None = None, rule: CollectionRule | None = None
    ) -> Collection:
        """Create a collection.

        Raises:
            NameInUseError: The workspace already has a collection with that name. Not an
                upsert: a collection is a deliberate object, and quietly returning somebody
                else's under the same name is how two people's sets become one.
        """
        ...

    async def get_collection(self, collection_id: str) -> Collection | None: ...

    async def find_collection(self, name: str) -> Collection | None:
        """Look one up by name, which is unique within a workspace."""
        ...

    async def list_collections(self) -> Sequence[Collection]: ...

    async def rename_collection(self, collection_id: str, name: str) -> Collection: ...

    async def describe_collection(self, collection_id: str, description: str | None) -> Collection:
        """Set or clear the description. ``None`` clears it."""
        ...

    async def set_collection_rule(
        self, collection_id: str, rule: CollectionRule | None
    ) -> Collection:
        """Attach, replace or remove the membership rule. ``None`` removes it.

        Three explicit verbs rather than one ``update`` taking optional fields, because an
        optional field cannot express the difference between "leave this alone" and "set this
        to nothing" — and for a rule those are the difference between a collection that keeps
        growing and one that stops.
        """
        ...

    async def delete_collection(self, collection_id: str) -> None:
        """Remove the collection and its memberships. **Never the documents.** Idempotent."""
        ...

    async def add_to_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        """Add documents by hand, returning how many memberships were new.

        Idempotent per document: adding one twice is not an error and does not double-count.

        Raises:
            ManiculeError: A document id belongs to another workspace or does not exist.
                Refused as a whole rather than partially applied, so a caller cannot be told
                a batch succeeded when part of it was dropped.
        """
        ...

    async def remove_from_collection(self, collection_id: str, document_ids: Sequence[str]) -> int:
        """Drop manual memberships, returning how many were removed.

        A document the *rule* selects stays in the collection, and that is not a bug to route
        around: the way to remove it is to change the rule.
        """
        ...

    async def collection_documents(
        self, collection_id: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        """Live members, manual and rule-selected together, newest first."""
        ...

    async def collections_for(self, document_id: str) -> Sequence[Collection]:
        """Every collection this document is in, by hand or by rule."""
        ...

    async def count_uncollected(self, *, source: str | None = None) -> int:
        """How many live documents no collection holds — the complement of every membership.

        The one question about collections whose answer is a property of the *workspace* rather
        than of a collection, and the reason it is here rather than derived by a caller: with no
        collections at all the answer is every document, and a caller computing it by summing
        :meth:`collection_documents` over :meth:`list_collections` would sum an empty list and
        report nothing wrong. It is counted, not listed, because the one surface that lists
        these documents also deletes them and the one that reports the number is ``doctor``.

        Args:
            source: Narrow to one connector's documents. ``None`` counts the workspace.
        """
        ...


@runtime_checkable
class TagStore(Protocol):
    """Arbitrary labels on documents, unique by name within a workspace."""

    async def ensure_tag(self, name: str, *, color: str | None = None) -> Tag:
        """Return the tag with this name, creating it if it is new.

        Idempotent, and the *only* way to make a tag. A separate strict ``create`` would exist
        to raise on a name that is already taken, which for a label is never the outcome the
        caller wanted — tagging is an operation people repeat.

        An existing tag's color is left alone. Overwriting it would make the last person to
        type the name the one who decides how it looks everywhere; :meth:`set_tag_color` is
        the deliberate way to change it.
        """
        ...

    async def get_tag(self, tag_id: str) -> Tag | None: ...

    async def find_tag(self, name: str) -> Tag | None: ...

    async def list_tags(self) -> Sequence[Tag]: ...

    async def rename_tag(self, tag_id: str, name: str) -> Tag:
        """Raises:
        NameInUseError: Another tag in this workspace already has that name. Merging the
            two would silently move every document from one label to another.
        """
        ...

    async def set_tag_color(self, tag_id: str, color: str | None) -> Tag: ...

    async def delete_tag(self, tag_id: str) -> None:
        """Remove the tag and every application of it. **Never the documents.** Idempotent."""
        ...

    async def tag_document(self, document_id: str, tag_ids: Sequence[str]) -> int:
        """Apply tags, returning how many applications were new.

        Raises:
            ManiculeError: The document or a tag belongs to another workspace, or does not
                exist.
        """
        ...

    async def untag_document(self, document_id: str, tag_ids: Sequence[str]) -> int: ...

    async def tags_for(self, document_id: str) -> Sequence[Tag]: ...

    async def documents_with_tags(
        self, tag_ids: Sequence[str], *, match_all: bool = False, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        """Live documents carrying any of these tags, or all of them when ``match_all``."""
        ...


@runtime_checkable
class VersionStore(Protocol):
    """A document's history across re-ingests, and what it means for a citation.

    **Nothing here writes a version.** History is recorded by the store, inside the same
    transaction that supersedes a document, because that is the only place both states are
    visible at once and the only place the write cannot be forgotten. A public ``record``
    would be a second author of the version sequence, and two authors of a monotonic counter
    is one too many.
    """

    async def list_versions(self, document_id: str) -> Sequence[DocumentVersion]:
        """Superseded states, oldest first. The current state is not among them."""
        ...

    async def get_version(self, document_id: str, version: int) -> DocumentVersion | None: ...

    async def current_version(self, document_id: str) -> int:
        """The ordinal of the state the document is in now: one past the highest stored row.

        ``1`` for a document that has never been superseded, and ``0`` for one this store
        cannot see.
        """
        ...

    async def resolve_citation(self, document_id: str, chunk_id: str) -> CitationResolution:
        """Say whether a stored citation still points at the text it named, and why not.

        Takes the document as well as the chunk because a dangling chunk id is opaque: it is a
        digest, it is not in ``chunks`` any more, and there is nothing to look it up against.
        Every citation manicule stores carries both.

        **A citation into a superseded version resolves to nothing, deliberately.** ``chunks.id``
        is derived from ``(document_id, position, text)``, so a chunk that survived a re-parse
        unchanged kept its id, and a chunk whose text changed did not — the id dangles instead
        of re-pointing. Returning the passage that replaced it would be a location that is
        plausible and wrong, which is the one outcome ``docs/contracts.md`` §1 rules out. What
        this returns instead is the absence, labeled: superseded, deleted, or unknown.
        """
        ...

    async def release_expired_versions(self, cutoff: datetime) -> int:
        """Let go of retained bytes for versions superseded before ``cutoff``.

        The history rows stay — they are a permanent record and cost almost nothing. What is
        released is ``original_ref``, which is the *capability* attached to a version rather
        than the record of it, and which pins a blob against garbage collection for as long as
        it is set. Without this, recording a version would grow the blob store without bound
        and quietly repeal the retention policy in ``docs/storage.md`` §7.

        Returns:
            How many versions let go of their bytes.
        """
        ...


@runtime_checkable
class TrashStore(Protocol):
    """Soft deletion, and the two ways back.

    Deletion is deferred, always (``docs/storage.md`` §8.2): soft-deleting sets a timestamp
    and touches neither the chunks, the vectors nor the lexical rows, all of which become
    invisible at the hydrating join. That is what makes a restore inside the grace period
    free — and what makes the scheduled sweep, which purges content once the grace period has
    passed, the thing a restore has to be designed around rather than against.
    """

    async def soft_delete_document(self, document_id: str) -> None:
        """Move a document to the trash. Idempotent, and it does not re-start the clock.

        A second soft delete of an already-deleted document leaves the original ``deleted_at``
        in place. Refreshing it would let a caller postpone the sweep indefinitely by deleting
        the same document repeatedly, which is a grace period that never expires.
        """
        ...

    async def restore_document(self, document_id: str) -> Restoration:
        """Take a document out of the trash, and say what that achieved.

        Two outcomes, and the caller has to be able to tell them apart. Inside the grace
        period the content is still there and clearing the timestamp puts the document back
        into service. Once the sweep has purged it, the row comes back holding no text: the
        result says so, and the repair is a single-document re-parse from retained bytes —
        rung 3 of the ladder, still not a re-crawl.
        """
        ...

    async def list_trash(
        self, *, grace_s: float, limit: int = 100, offset: int = 0
    ) -> Sequence[TrashEntry]:
        """What is in the trash, longest-deleted first, with how long each has left.

        ``grace_s`` is passed rather than known, because the sweep that will act on it takes
        the same number from configuration and a second copy of a policy is a second thing to
        get wrong.
        """
        ...

    async def delete_document(self, document_id: str) -> None:
        """Hard-delete a document and everything hanging off it. Idempotent.

        The cascade reaches ``chunks``, whose delete trigger clears the lexical rows and
        writes vector tombstones, so the derived stores are cleaned by the database rather
        than by whichever caller remembered to.
        """
        ...


@runtime_checkable
class ChunkRelationStore(Protocol):
    """Typed edges between chunks: parent links and sibling links.

    Edges are stored **once and read from both ends**. ``docs/storage.md`` §4.4 indexes
    ``target_chunk_id`` for exactly that reason — lookups are ``WHERE source = ? OR target = ?``
    and a composite primary key leading with ``source`` cannot serve the second half — so a
    mirror row would double the table to answer a query the schema already answers, and would
    create a pair that can fall out of step.

    Both columns are real foreign keys with ``ON DELETE CASCADE``, so an edge cannot outlive
    either chunk. That is the payoff of chunks being a table rather than metadata inside a
    vector store: orphan cleanup is the cascade, not a pattern match over formatted ids.
    """

    async def relate(
        self, source_chunk_id: str, target_chunk_id: str, relation_type: ChunkRelationType
    ) -> None:
        """Record an edge. Idempotent — the same edge twice is one row.

        Raises:
            ManiculeError: Either chunk is outside this workspace or does not exist, or the
                two are the same chunk. A cross-workspace edge is the sharper of the two: it
                would let a lookup from one tenant's chunk return another tenant's chunk id.
        """
        ...

    async def unrelate(
        self, source_chunk_id: str, target_chunk_id: str, relation_type: ChunkRelationType
    ) -> None:
        """Remove an edge, in the direction it was written. Idempotent."""
        ...

    async def related(
        self, chunk_id: str, *, types: AbstractSet[ChunkRelationType] = frozenset()
    ) -> Sequence[ChunkEdge]:
        """Every edge touching this chunk, from either end, optionally filtered by type.

        An empty ``types`` restricts nothing, exactly as an empty field on
        :class:`~manicule.core.retrieval.Filter` does. There is deliberately no ``None``
        alongside it: two spellings of "no restriction" is one too many, and the second one is
        always the one somebody reads as its opposite.
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

       **Locked.** The evaluation harness exists (:mod:`manicule.evaluation`), and every
       preference record it writes names the stage list that produced each side. Widening the
       signature now invalidates every recorded result, because a stage that receives more can
       no longer be replayed against a run that gave it less.

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

    context_window: int
    """Tokens this generator will actually accept, prompt and completion together.

    **Served, not advertised.** Not the model's trained maximum: the window that will be in
    force for this configuration. Ollama applies a runtime ``num_ctx`` that defaults far below
    what modern models are trained for, and a prompt exceeding it is truncated from the front
    rather than refused — which silently discards the system prompt and the citation protocol,
    and presents as a model that does not follow instructions. An Ollama-backed implementation
    therefore derives this from ``/api/show`` combined with the ``num_ctx`` manicule itself
    sets; a hosted one reads its library's model metadata.

    Read by the startup cross-check in ``docs/retrieval.md`` §7.4:
    ``context_tokens + history_tokens + system_prompt_tokens + generation_reserve`` must fit
    it. A profile that does not fit is a refusal with both numbers named, because a limit that
    can only be discovered by exceeding it gets discovered in production.
    """

    def generate(self, query: Query, context: Context) -> AsyncIterator[Token]:
        """Stream the answer. The final token carries a finish reason.

        Iterate it through :func:`generating`, never bare. What an abandoned generation
        stream holds is worse than a file handle: an open HTTP response to a model that is
        still working. On a hosted provider that is tokens being billed for an answer nobody
        will read; on a local one the model keeps generating until the client goes away, so a
        user who closes a tab and asks again is queued behind their own abandoned answer.
        """
        ...


def _accepted(generator: Generator, extra: Mapping[str, object] | None) -> dict[str, object]:
    """The subset of ``extra`` this generator's ``generate`` actually declares.

    Silently dropping the rest is right here and would be wrong elsewhere: these are inputs a
    protocol-conformant generator is *entitled* not to have, and the answer path records which
    ones it took, so a conversation that did not reach the model is a fact in the trace rather
    than a silent loss.
    """
    if not extra:
        return {}
    try:
        parameters = inspect.signature(generator.generate).parameters
    except (TypeError, ValueError):  # pragma: no cover - a callable with no readable shape
        return {}
    declared = {
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(extra)
    return {name: value for name, value in extra.items() if name in declared}


CLOSE_DEADLINE_S = 5.0
"""How long :func:`generating` waits for a provider connection to close.

Past it the connection is abandoned to the pool's own teardown. The trade is deliberate and
one-directional: a leaked socket is recovered by the pool, while a shutdown that blocks
indefinitely on a misbehaving remote server is recovered by nothing.
"""


@asynccontextmanager
async def generating(
    generator: Generator,
    query: Query,
    context: Context,
    *,
    close_deadline_s: float | None = CLOSE_DEADLINE_S,
    extra: Mapping[str, object] | None = None,
) -> AsyncGenerator[AsyncIterator[Token]]:
    """Iterate a generator's tokens, closing the stream on **every** exit path.

    :func:`parsing`'s sibling, and it exists for the same reason with a worse resource behind
    it (:meth:`Generator.generate`). Use it the same way::

        async with generating(generator, query, context) as tokens:
            async for token in tokens:
                ...

    Two exits have to be covered and they arrive differently. ``aclose()`` — a consumer that
    stopped early, or this ``finally`` — raises :exc:`GeneratorExit` at the ``yield``.
    :exc:`asyncio.CancelledError` — the client disconnected and the task was canceled —
    arrives at whatever ``await`` the generator is suspended on, usually inside the provider
    read. One ``try``/``finally`` in the implementation covers both, and only a ``finally``
    runs after ``GeneratorExit``.

    Two rules on what may go in that ``finally``, both of which this contract relies on.
    **Cleanup awaits nothing unbounded**, because an ``await`` that never completes hangs
    ``aclose()``, which is itself being awaited by somebody else's ``finally``;
    ``close_deadline_s`` is the backstop for an implementation that breaks the rule.
    **Cleanup never yields**: yielding after :exc:`GeneratorExit` raises ``RuntimeError:
    async generator ignored GeneratorExit``, so the final token carrying a finish reason is
    emitted on the normal path only. A stream nobody is reading has nobody to tell.

    ``extra`` forwards keyword arguments the bound generator declares **beyond** the
    protocol — and *only* those it declares. Forwarding unconditionally made this the opposite
    of what it says: a generator that did not accept a key raised ``TypeError`` before
    streaming began, so the mechanism for optional extras was a mechanism for mandatory
    ones.

    The protocol fixes two inputs and conversation history is a third the seam has
    no channel for; widening :meth:`Generator.generate` would be a change to a contract three
    implementations are written against, so the additional inputs travel as optional keywords
    instead — which :func:`manicule.testing.assert_protocol_signatures` already sanctions,
    since a caller working from the protocol never passes them. A generator that declares
    none is called with none, and the answer path records that it did rather than letting a
    conversation quietly vanish.
    """
    stream = generator.generate(query, context, **_accepted(generator, extra))
    try:
        yield stream
    finally:
        await aclose(stream, timeout=close_deadline_s)


# --- connectors --------------------------------------------------------------------------


@runtime_checkable
class Connector(Protocol):
    """Finds documents in a source and fetches them."""

    name: str

    @property
    def watermark(self) -> Watermark | None:
        """How far the **last completed** enumeration got, or ``None`` if none has.

        The other half of :meth:`discover`, which consumes a watermark and — without this —
        had nowhere to produce the next one, leaving ``connectors.watermark``
        (``storage.md`` §4.7) unfillable by anything working through this protocol.

        **It is safe to persist if and only if every document ``discover`` has yielded has
        been durably committed.** That is a condition on the caller, not on the connector: the
        connector promises only that this reflects a *complete* enumeration, never a partial
        one, and the caller must not store it until what that enumeration produced is stored
        too. Read it after a clean run and not otherwise.

        Read-only, so an implementation is free to expose a stored value or a computed one.

        .. warning::

           **Persisting a watermark for work that was not committed loses documents
           permanently.** The next sync starts from the stored position, so anything the
           interrupted run enumerated but did not store is never enumerated again. Not
           delayed — invisible, with nothing raised and nothing to notice, until somebody
           searches for a document that has been in the source all along and is not in the
           index. It is the same class of failure as a citation pointing at a page that does
           not exist: internally consistent, quietly wrong, and undetectable from the inside.

        The race between "yielded" and "committed" is real and is deliberately not eliminated.
        It is made survivable instead, by connectors that overlap their queries slightly rather
        than resuming exactly — the Confluence connector reaches five minutes back before its
        stored position (``docs/connectors/confluence.md`` §2), because re-enumerating a small
        overlap costs a version comparison that change detection was going to make anyway and
        content-hash dedup absorbs the rest. Pretending the race is not there is what turns it
        into lost documents.
        """
        ...

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


@runtime_checkable
class BatchedDiscoveryConnector(Protocol):
    """Optionally exposes source-native discovery boundaries to durable consumers.

    A batch is one bounded source response, not an arbitrary process-sized buffer.  Durable
    ingest can therefore commit every record in that response in one transaction before asking
    the connector to follow its next cursor.  ``Connector.discover`` remains the mandatory
    compatibility surface; connectors without expiring page cursors need not implement this.
    """

    def discover_batches(
        self, watermark: Watermark | None
    ) -> AsyncIterator[Sequence[DiscoveredDoc]]: ...


@runtime_checkable
class BatchedReconciliationConnector(Protocol):
    """Optionally preserves source-native cursor pages during deletion enumeration."""

    def reconcile_batches(self) -> AsyncIterator[Sequence[SourceId]]: ...


@runtime_checkable
class EnumerationProgressConnector(Protocol):
    """Optionally exposes aggregate-only live enumeration facts for durable diagnostics.

    ``None`` means the current operation has nothing adaptive to report — an incremental
    query, or a connector that never changes its request shape. Durable ingest persists a
    non-``None`` snapshot on the acquisition run whenever it changes, so status surfaces can
    distinguish a walk that is adapting to source latency from one that is hung, without the
    connector ever touching storage.
    """

    def enumeration_progress(self) -> EnumerationProgress | None: ...


# --- middleware --------------------------------------------------------------------------


@runtime_checkable
class Middleware(Protocol):
    """Observes and transforms documents as they pass through ingest.

    Every hook is **transformational**: it receives a value and returns the value the
    pipeline continues with. There is no in-place mutation contract and no discarded return
    — a hook that returns a new object has an effect, which is the only behavior anybody
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

    **A hook may also carry a ``version`` string**, and it is recorded beside its name as
    ``name@version`` — in the chunk fingerprint for a hook declaring
    :attr:`mutates_embedded_text`, and in the glossary and relation lineage for every hook. Set
    it when something that moves independently of the hook's own code decides what it writes:
    the diagram middleware names the tree-sitter runtime and grammar pack it reads diagrams with,
    because nothing else records them for the chunks it rewrites. It is read if present rather
    than declared here, because a protocol member would make every middleware already written
    against this protocol structurally stop satisfying it; absent, it records ``name@``, which
    is what every middleware recorded before the attribute existed.
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


@runtime_checkable
class ChunkRelationExtractor(Protocol):
    """A middleware that derives :class:`~manicule.core.organization.ChunkEdge` rows.

    Implemented **in addition to** :class:`Middleware` rather than instead of it. The extraction
    itself is an ordinary ``after_store`` hook writing through a
    :class:`ChunkRelationStore`; what this protocol adds is the one thing a hook cannot be asked
    for through the middleware interface — an account of the rules it derived those edges under.

    **Why the pipeline has to be able to ask.** Edges are derived state, and derived state whose
    producer is not versioned is state that reports itself current after the producer has been
    corrected. :class:`~manicule.core.fingerprints.RelationFingerprint` is what fixes that, and
    it is recorded per document by the pipeline rather than by the hook — because the pipeline is
    what knows whether the hook chain ran to the end, and because the identity has to include
    the whole configured chain, which a hook cannot see.

    So an extractor states its own half and the pipeline adds the rest. A middleware that does
    not implement this is not an extractor and contributes nothing to the fingerprint; an
    installation with none configured records
    :meth:`~manicule.core.fingerprints.RelationFingerprint.disabled`.
    """

    def relation_rules(self) -> RelationRules:
        """What decides the edges this extractor writes. Never awaits; never reads the index."""
        ...


__all__ = [
    "CLOSE_DEADLINE_S",
    "AdoptingVectorStore",
    "AnnIndexMaintenance",
    "ChunkRelationExtractor",
    "ChunkRelationStore",
    "Chunker",
    "CollectionStore",
    "Connector",
    "DocStore",
    "Embedder",
    "Generator",
    "InspectableVectorStore",
    "Middleware",
    "MultiWorkspaceVectorStore",
    "Parser",
    "PublicationAwareVectorStore",
    "PublicationBoundVectorStore",
    "Reranker",
    "ResettableVectorStore",
    "RetrievalStage",
    "TagStore",
    "TokenStateEmbedder",
    "TrashStore",
    "VectorIntegrityMaintenance",
    "VectorStore",
    "VersionStore",
    "aclose",
    "bounded",
    "generating",
    "parsing",
    "read_blocks",
]
