"""Conformance suites: the obligations an implementation must meet, as runnable checks.

Shipped rather than kept in the test tree, so a third-party plugin can be held to the same
standard as a built-in one with an import instead of a copied file.

Every check here corresponds to a promise made in :mod:`manicule.core.protocols`. A promise
in a docstring is a hope; the same promise with a function that fails when it is broken is a
contract.

Checks raise :class:`AssertionError` explicitly rather than using ``assert``, so they keep
working under ``python -O``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence

from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, Document, ParsedBlock, RawDocument
from manicule.core.embedding import EmbedFingerprint, Pooling, Vector
from manicule.core.errors import ContextOverflowError, FingerprintMismatchError
from manicule.core.protocols import (
    Chunker,
    Connector,
    Embedder,
    Parser,
    RetrievalStage,
    VectorStore,
)
from manicule.core.retrieval import Candidate, Query


def _fail(message: str) -> None:
    raise AssertionError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _normalise(text: str) -> str:
    """Collapse whitespace, which parsers legitimately reflow."""
    return " ".join(text.split())


# --- protocol shape ---------------------------------------------------------------------


def assert_protocol_signatures(implementation: object, protocol: type) -> None:
    """Check that an implementation's method *signatures* match the protocol's.

    ``@runtime_checkable`` deliberately checks only that attributes exist — never their
    signatures — so ``isinstance(store, DocStore)`` returns ``True`` for an implementation
    whose parameter is called ``text_query`` where the protocol says ``text``. The mismatch
    surfaces at the first keyword call, which may be in a caller written months later, and the
    ``isinstance`` check that was supposed to catch it reported success the whole time.

    This is what makes a structural protocol safe to rely on: every method the protocol
    declares must exist, be callable, and accept the same parameters in the same order under
    the same names. An implementation may add parameters the protocol does not have, provided
    they have defaults — a caller working from the protocol will never pass them.

    Args:
        implementation: The object or class under test.
        protocol: The :class:`typing.Protocol` it claims to satisfy.

    Raises:
        AssertionError: A method is missing, is not callable, or has a different signature.
    """
    import inspect  # noqa: PLC0415 - only this check needs it

    for name, declared in vars(protocol).items():
        if name.startswith("_") or not inspect.isfunction(declared):
            continue

        found = getattr(implementation, name, None)
        _require(found is not None, f"{protocol.__name__}.{name} is not implemented")
        _require(callable(found), f"{protocol.__name__}.{name} is not callable")

        expected = _parameters(inspect.signature(declared))
        actual = _parameters(inspect.signature(found))  # pyright: ignore[reportArgumentType] - callable, checked above

        _require(
            actual[: len(expected)] == expected,
            f"{protocol.__name__}.{name} takes {expected} but the implementation takes "
            f"{actual}. isinstance() cannot see this: @runtime_checkable checks that the "
            f"attribute exists, not what it accepts, so a keyword call fails at run time "
            f"against a check that passed",
        )
        for extra in actual[len(expected) :]:
            _require(
                extra.endswith("="),
                f"{protocol.__name__}.{name} has an extra required parameter {extra!r}; a "
                f"caller working from the protocol will never pass it",
            )


def _parameters(signature: object) -> list[str]:
    """Parameter names in order, with ``=`` appended to those carrying a default.

    ``self`` is dropped: it is present on the protocol's unbound function and absent from a
    bound method, and the difference says nothing about compatibility.
    """
    import inspect  # noqa: PLC0415

    if not isinstance(signature, inspect.Signature):  # pragma: no cover - defensive
        return []
    names: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        suffix = "=" if parameter.default is not inspect.Parameter.empty else ""
        names.append(f"{parameter.name}{suffix}")
    return names


# --- parsers ---------------------------------------------------------------------------


async def assert_parser_contract(parser: Parser, raw: RawDocument) -> list[ParsedBlock]:
    """Check a parser's promises against one document, and return what it produced.

    Verifies the round-trip obligation: resolving each block's anchor returns the text the
    block claims. This is the check that stops citations pointing at places that do not
    exist — the failure mode is not a crash but a plausible wrong number, so it has to be
    tested rather than reviewed.
    """
    _require(parser.media_types, "parser declares no media types, so nothing routes to it")

    blocks = [block async for block in parser.parse(raw)]

    for index, block in enumerate(blocks):
        where = f"block {index}"
        _require(block.text, f"{where}: empty text; a block with no text is not a block")

        resolved = await parser.resolve(block.anchor, raw)

        if isinstance(block.anchor, Unlocated):
            _require(
                resolved is None,
                f"{where}: the anchor is Unlocated but resolve() returned text, so the "
                f"anchor claims less than the parser knows",
            )
            _require(block.anchor.reason, f"{where}: Unlocated without a reason")
            continue

        _require(
            resolved is not None,
            f"{where}: anchor {block.anchor!r} does not resolve. An anchor that cannot be "
            f"resolved is a citation that cannot be checked; emit Unlocated instead",
        )
        _require(
            _normalise(block.text) in _normalise(resolved or ""),
            f"{where}: the anchor resolves to text the block does not claim.\n"
            f"  block:    {block.text[:120]!r}\n"
            f"  resolved: {(resolved or '')[:120]!r}",
        )

    return blocks


# --- chunkers --------------------------------------------------------------------------


def assert_chunker_contract(
    chunker: Chunker,
    document: Document,
    blocks: Iterable[ParsedBlock],
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Check a chunker's promises, and return what it produced.

    Pass ``embedder`` to also check the budget: a chunk budget above the embedder's
    effective sequence length is silently truncated at embed time, and the chunk is then
    indexed as its opening tokens while still claiming all of its text.
    """
    block_list = list(blocks)

    if embedder is not None:
        limit = embedder.fingerprint.max_sequence_length
        _require(
            chunker.fingerprint.max_tokens <= limit,
            f"chunk budget is {chunker.fingerprint.max_tokens} tokens but the embedder "
            f"attends to {limit}. Everything past the limit is dropped without an error, so "
            f"the chunk would claim text the index never saw",
        )
        _require(
            chunker.fingerprint.tokenizer_id == embedder.fingerprint.tokenizer_id,
            f"the chunker counts tokens with {chunker.fingerprint.tokenizer_id!r} and the "
            f"embedder uses {embedder.fingerprint.tokenizer_id!r}. A budget measured with "
            f"the wrong vocabulary is not a budget",
        )

    chunks = chunker.chunk(document, block_list)

    seen: set[str] = set()
    for position, chunk in enumerate(chunks):
        where = f"chunk {position}"
        _require(chunk.document_id == document.id, f"{where}: belongs to a different document")
        _require(chunk.text, f"{where}: empty text")
        _require(chunk.id not in seen, f"{where}: duplicate id {chunk.id!r}")
        seen.add(chunk.id)
        _require(
            chunk.position == position,
            f"{where}: position is {chunk.position}, breaking document order",
        )
        _require(
            _normalise(chunk.text) in _normalise(chunk.embed_text),
            f"{where}: embed_text does not contain text. embed_text carries the heading "
            f"breadcrumb *and* the text; dropping the text embeds the scaffolding alone",
        )
        _require(
            chunk.token_count > 0,
            f"{where}: token_count is 0, so context fitting cannot account for it",
        )

    if any(block.text.strip() for block in block_list):
        _require(chunks, "blocks with text produced no chunks")
    return chunks


# --- embedders -------------------------------------------------------------------------


async def assert_embedder_contract(embedder: Embedder, texts: Sequence[str] | None = None) -> None:
    """Check that an embedder's output matches the identity it advertises.

    The dimension is taken from the embedder, never assumed here. That is the point: a check
    that knew the number would be a second place for it to be wrong.
    """
    sample = list(texts) if texts is not None else ["the first text", "a second, longer one"]
    fingerprint = embedder.fingerprint

    _require(fingerprint.dimension > 0, "fingerprint.dimension must be positive")
    _require(fingerprint.model_id, "fingerprint.model_id must identify the model")

    vectors = await embedder.embed(sample)
    _require(
        len(vectors) == len(sample),
        f"embed() returned {len(vectors)} vectors for {len(sample)} texts",
    )
    for index, vector in enumerate(vectors):
        _require(
            len(vector) == fingerprint.dimension,
            f"vector {index} has {len(vector)} dimensions, but the fingerprint says "
            f"{fingerprint.dimension}. The fingerprint is what the index is built against, "
            f"so a disagreement here corrupts every later search",
        )

    empty = await embedder.embed([])
    _require(list(empty) == [], "embed([]) must return an empty list, not a batch of nothing")


async def assert_refuses_oversized_chunks(
    embed_batch: Callable[[Sequence[Chunk]], Awaitable[object]], embedder: Embedder
) -> None:
    """Check that a path which embeds stored chunks refuses ones the model cannot read.

    Aimed squarely at **re-embed**, which is where this goes wrong. Re-embedding reads
    stored ``embed_text`` and does not re-chunk, so the chunker's own budget refusal never
    runs. If the model is later reconfigured to a shorter sequence length — a different
    checkpoint, an edited config, a changed backend default — the embedding fingerprint is
    unchanged, no comparison fires, and every oversized chunk is silently truncated into a
    vector claiming text it never saw. Across a corpus, in one command, with no error.

    Pass the function that embeds a batch of stored chunks. It must call
    :func:`manicule.core.embedding.require_within_context` — or do the equivalent check
    itself — before handing anything to the model.

    Args:
        embed_batch: The path under test. Called with one chunk that does not fit.
        embedder: Whose ``max_sequence_length`` the chunk is built to exceed.
    """
    limit = embedder.fingerprint.max_sequence_length
    oversized = Chunk(
        id="oversized-by-one",
        document_id="doc",
        text="x",
        embed_text="x",
        anchor=Unlocated(reason="synthetic chunk for a contract check"),
        position=0,
        token_count=limit + 1,
    )

    try:
        await embed_batch([oversized])
    except ContextOverflowError:
        return
    _fail(
        f"a chunk of {limit + 1} tokens was embedded by a model that reads {limit}. "
        f"Everything past the limit is dropped without an error, so the stored vector "
        f"describes an opening fragment while the chunk still claims all of its text. "
        f"Call require_within_context() before embedding stored chunks"
    )


# --- vector stores ---------------------------------------------------------------------


async def assert_vector_store_is_dimension_agnostic(
    make_store: Callable[[], VectorStore], chunks: Sequence[Chunk]
) -> None:
    """Check that a store works at whatever dimension the embedder happens to report.

    Run at two deliberately unusual dimensions. Any implementation carrying a hardcoded
    dimension — in a schema, a buffer size, an assertion — fails at least one of them, which
    turns "never hardcode the dimension" from an instruction into something the build
    enforces.
    """
    for dimension in (7, 13):
        fingerprint = _fingerprint(dimension)
        store = make_store()
        await store.ensure_ready(fingerprint)

        stored = await store.fingerprint()
        _require(
            stored is not None and stored.matches(fingerprint),
            f"the store did not record the fingerprint it was prepared with at d={dimension}",
        )

        vectors: list[Vector] = [[float(i)] * dimension for i, _ in enumerate(chunks)]
        await store.upsert(list(chunks), vectors)
        _require(
            await store.count() == len(chunks),
            f"the store holds the wrong number of vectors at d={dimension}",
        )

        results = await store.search([0.0] * dimension, k=len(chunks))
        _require(results, f"search returned nothing at d={dimension}")
        _require(
            all(result.chunk.text for result in results),
            "search must return Candidates carrying the chunk, not bare ids: a reranker "
            "cannot score an id, and refetching per stage turns one query into N",
        )


async def assert_vector_store_rejects_foreign_vectors(
    make_store: Callable[[], VectorStore],
) -> None:
    """Check that a store refuses vectors from a different model.

    Including a different model of the *same* dimension, which is the case a size check
    misses and the one that quietly ruins retrieval: the writes succeed, the searches
    return, and every answer is drawn from a space the query does not live in.
    """
    store = make_store()
    await store.ensure_ready(_fingerprint(8, model_id="model-a"))

    try:
        await store.ensure_ready(_fingerprint(8, model_id="model-b"))
    except FingerprintMismatchError:
        return
    _fail(
        "the store accepted a second model with the same dimension. Vectors from two models "
        "are not comparable, and nothing downstream can detect that they were mixed"
    )


def _fingerprint(dimension: int, model_id: str = "test/model") -> EmbedFingerprint:
    return EmbedFingerprint(
        model_id=model_id,
        dimension=dimension,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=512,
    )


# --- retrieval stages ------------------------------------------------------------------


async def assert_retrieval_stage_contract(
    stage: RetrievalStage, query: Query, candidates: Sequence[Candidate]
) -> list[Candidate]:
    """Check that a stage is a stage: candidates in, candidates out, input untouched.

    Uniformity is what lets the evaluation harness compare pipelines by configuration. A
    stage that mutates its input makes the pipeline order-dependent, and an order-dependent
    pipeline cannot be compared with another one.
    """
    _require(stage.name, "stage has no name, so configuration cannot select it")

    given = list(candidates)
    before = [candidate.model_copy(deep=True) for candidate in given]
    produced = await stage.run(query, given)

    _require(
        produced is not given,
        f"stage {stage.name!r} returned the very list it was given; a stage produces a new "
        f"list, so that a pipeline can be replayed stage by stage",
    )
    _require(
        given == before,
        f"stage {stage.name!r} mutated the candidates it was given; return new ones instead",
    )
    for candidate in produced:
        _require(candidate.chunk.id, f"stage {stage.name!r} produced a candidate with no chunk id")
    return produced


# --- connectors ------------------------------------------------------------------------


async def assert_connector_contract(connector: Connector) -> None:
    """Check the three things every connector must do, including the one easy to skip.

    ``reconcile`` is part of the protocol because incremental sync cannot detect deletions.
    A connector returning nothing from it is claiming its source has no documents, which
    would delete the lot — so it must yield what exists, and this check makes an omission
    visible rather than catastrophic.
    """
    _require(connector.name, "connector has no name")

    discovered = [doc async for doc in connector.discover(None)]
    for doc in discovered:
        _require(doc.ref.source_id, "a discovered document has no source id")
        _require(doc.ref.uri, "a discovered document has no uri")

    if discovered:
        raw = await connector.fetch(discovered[0].ref)
        _require(
            raw.source_id == discovered[0].ref.source_id,
            "fetch() returned a document with a different source id than it was asked for",
        )
        _require(raw.media_type, "the fetched document has no media type")

    reconciled = {source_id async for source_id in connector.reconcile()}
    missing = {doc.ref.source_id for doc in discovered} - reconciled
    _require(
        not missing,
        f"reconcile() omitted {sorted(missing)}, which discover() had just returned. "
        f"Reconciliation drives deletion, so anything it omits is deleted from the index",
    )


__all__ = [
    "assert_chunker_contract",
    "assert_connector_contract",
    "assert_embedder_contract",
    "assert_parser_contract",
    "assert_protocol_signatures",
    "assert_refuses_oversized_chunks",
    "assert_retrieval_stage_contract",
    "assert_vector_store_is_dimension_agnostic",
    "assert_vector_store_rejects_foreign_vectors",
]
