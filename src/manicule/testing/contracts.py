"""Conformance suites: the obligations an implementation must meet, as runnable checks.

Shipped rather than kept in the test tree, so a third-party plugin can be held to the same
standard as a built-in one with an import instead of a copied file.

Every check here corresponds to a promise made in :mod:`manicule.core.protocols` — or, for the
last two, in :mod:`manicule.config` and by the stores that carry the workspace boundary. A
promise in a docstring is a hope; the same promise with a function that fails when it is broken
is a contract.

Checks raise :class:`AssertionError` explicitly rather than using ``assert``, so they keep
working under ``python -O``.
"""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, Document, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.embedding import EmbedFingerprint, Pooling, Vector
from manicule.core.errors import ContextOverflowError, FingerprintMismatchError
from manicule.core.protocols import (
    Chunker,
    Connector,
    DocStore,
    Embedder,
    Middleware,
    Parser,
    RetrievalStage,
    VectorStore,
    read_blocks,
)
from manicule.core.retrieval import Candidate, Query

if TYPE_CHECKING:  # pragma: no cover - imported for typing only, never at run time
    from manicule.config.settings import Settings


@asynccontextmanager
async def closing[T](iterator: AsyncIterator[T]) -> AsyncGenerator[AsyncIterator[T]]:
    """Consume an async iterator and close it deterministically if it can be closed.

    An async *generator* suspended at a ``yield`` holds whatever it had open — a database
    session, a file, an HTTP response — until it is finalised. Drained bare, that finalisation
    happens at garbage-collection time through the event loop's async-generator hook, possibly
    after the loop it belongs to has closed. That is a resource leak at best and an
    interpreter crash at worst; it has been observed to segfault CPython 3.13.

    :func:`contextlib.aclosing` is the standard answer but requires an ``AsyncGenerator``,
    while every protocol in :mod:`manicule.core.protocols` declares the wider
    ``AsyncIterator`` — deliberately, so an implementation is free to return a hand-written
    iterator. This closes what can be closed and leaves the rest alone, so the protocol does
    not have to narrow.
    """
    try:
        yield iterator
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


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

    blocks = await read_blocks(parser, raw)

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
    """Check the four things every connector must do, including the two easy to skip.

    ``reconcile`` is part of the protocol because incremental sync cannot detect deletions.
    A connector returning nothing from it is claiming its source has no documents, which
    would delete the lot — so it must yield what exists, and this check makes an omission
    visible rather than catastrophic.

    ``watermark`` is the other one, and its failure is quieter still: see
    :func:`_assert_watermark_survives_abandonment`.
    """
    _require(connector.name, "connector has no name")

    await _assert_watermark_survives_abandonment(connector)

    async with closing(connector.discover(None)) as discovered_iter:
        discovered = [doc async for doc in discovered_iter]
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

    async with closing(connector.reconcile()) as reconcile_iter:
        reconciled = {source_id async for source_id in reconcile_iter}
    missing = {doc.ref.source_id for doc in discovered} - reconciled
    _require(
        not missing,
        f"reconcile() omitted {sorted(missing)}, which discover() had just returned. "
        f"Reconciliation drives deletion, so anything it omits is deleted from the index",
    )


async def _assert_watermark_survives_abandonment(connector: Connector) -> None:
    """Check that an abandoned enumeration does not move the watermark.

    The failure this exists to catch is a connector that advances its watermark **as it
    yields** rather than when the enumeration completes. Nothing about that looks wrong: every
    document it produced is correct, the run reports success, and the watermark it offers is a
    real position in the source. But if the run was interrupted — cancellation, an error three
    documents later, a bounded queue that was never drained — that position is *past* documents
    the caller never received. The next sync starts from it, so those documents are never
    enumerated again. Not delayed: permanently invisible, with nothing raised and nothing to
    notice, until somebody searches for something that has been in the source the whole time.

    So: take one document, abandon the stream, and require that the watermark has not moved.
    ``None`` passes, because a connector offering nothing is offering nothing wrong; what fails
    is a watermark that advanced on the strength of a walk that did not finish.

    A source with no documents at all is skipped rather than passed — there is nothing an
    abandoned enumeration could have been abandoned part-way through.

    **This is one of two checks on the same guarantee, and it is not redundant with the other.**
    The ingest pipeline separately refuses to persist a watermark after a run that did not
    finish. That gate fires on a run which has already gone wrong; this one fires on a run where
    nothing went wrong at all, which is the only kind anyone looks at. A connector can pass this
    check and still lose documents through a caller that ignores the other. Deleting either
    restores the same failure — documents in the source, enumerated once, in no index,
    permanently.
    """
    before = connector.watermark

    async with closing(connector.discover(None)) as stream:
        try:
            await anext(stream)
        except StopAsyncIteration:
            return

    after = connector.watermark
    _require(
        after is None or after == before,
        f"the watermark moved to {after!r} after an enumeration that was abandoned after one "
        f"document. A watermark advanced by a walk that did not finish is a position past "
        f"documents nobody received: the next sync starts there and never sees them again, "
        f"with no error and nothing to notice. Advance it when the enumeration completes, and "
        f"leave the caller to decide whether the run was clean",
    )


# --- middleware ------------------------------------------------------------------------


async def assert_middleware_contract(
    middleware: Middleware,
    document: Document,
    chunks: Sequence[Chunk],
    blocks: Sequence[ParsedBlock] | None = None,
) -> list[Chunk]:
    """Check that a middleware leaves ``Chunk.text`` alone, and returns what it produced.

    This is the check no parser suite can perform. :func:`assert_parser_contract` verifies
    that resolving a chunk's anchor returns the text the chunk claims — but it runs against
    the parser, before any middleware exists. A hook that rewrites ``text`` — on a block in
    ``after_parse`` or a chunk in ``after_chunk``, which are the same corruption — breaks that
    correspondence *afterwards*, so every parser in the project still passes while the
    corpus grows citations quoting text no source document contains.

    ``embed_text`` is deliberately not checked for equality: rewriting it is the reason
    ``after_chunk`` exists. What is checked is that a middleware which rewrites it says so,
    because the ingest pipeline folds that declaration into the chunk fingerprint and a
    silent mutation produces a corpus no fingerprint describes.

    Args:
        middleware: The implementation under test.
        document: Passed through to the hook.
        chunks: Input chunks. Give at least one with a non-trivial ``text``.
        blocks: Input blocks for the ``after_parse`` hook. Optional only because a
            middleware may not implement it; pass them whenever it does.

    Returns:
        The chunks the middleware returned, so a caller can make further assertions.
    """
    _require(chunks, "assert_middleware_contract needs at least one chunk to have an opinion")

    if blocks:
        before_blocks = [block.text for block in blocks]
        returned_blocks = await middleware.after_parse(document, list(blocks))
        for index, block in enumerate(returned_blocks):
            if index < len(before_blocks) and block.text != before_blocks[index]:
                _fail(
                    f"middleware {middleware.name!r} rewrote ParsedBlock.text at index "
                    f"{index}. Block text becomes chunk text, so this corrupts a citation "
                    f"exactly as rewriting Chunk.text does — the block's anchor still points "
                    f"at source text that no longer matches what the block claims. Reshape "
                    f"embed_text in after_chunk instead"
                )

    before = {chunk.id: chunk.text for chunk in chunks}
    before_embed = {chunk.id: chunk.embed_text for chunk in chunks}

    returned = await middleware.after_chunk(document, list(chunks))

    for chunk in returned:
        original = before.get(chunk.id)
        if original is None:
            continue
        if chunk.text != original:
            _fail(
                f"middleware {middleware.name!r} rewrote Chunk.text on {chunk.id!r}. "
                f"text is what a citation displays and what resolving an anchor must "
                f"reproduce, so changing it here breaks the round-trip guarantee after "
                f"every parser test has already passed — the corpus stays internally "
                f"consistent while its citations quote text the source does not contain. "
                f"Rewrite embed_text instead and declare mutates_embedded_text"
            )

    mutated_embed = any(
        chunk.embed_text != before_embed.get(chunk.id, chunk.embed_text) for chunk in returned
    )
    if mutated_embed and not middleware.mutates_embedded_text:
        _fail(
            f"middleware {middleware.name!r} rewrote Chunk.embed_text without declaring "
            f"mutates_embedded_text. That changes every vector while both fingerprint "
            f"refusals pass, because neither fingerprint knows middleware exists. Set "
            f"mutates_embedded_text = True so the pipeline can fold it into the chunk "
            f"fingerprint"
        )

    return returned


# --- the workspace boundary --------------------------------------------------------------


async def assert_pipeline_enforces_scope(
    pipeline: Sequence[RetrievalStage],
    docstore: DocStore,
    query: Query,
    *,
    expect_results: bool = True,
) -> list[Candidate]:
    """Check that no stage of a pipeline emits a chunk the query's scope excludes.

    **This is the check that makes one deliberate omission safe.**
    :data:`manicule.storage.vectors.EXEMPT_FILTER_FIELDS` lets the vector store drop
    ``workspace_ids`` — it has no column for tenancy and will not get one, because copying a
    value into a derived store creates a value that can disagree. The boundary moves to the
    hydrating join inside the dense stage (``docs/retrieval.md`` §4.2) rather than
    disappearing, and "it is enforced somewhere else" is a claim, not a guarantee, until
    something fails when it is not.

    Run it against a fixture that holds what a leak would draw from: chunks of a soft-deleted
    document, chunks of one that is not ``indexed``, and chunks belonging to another
    workspace. A dense stage that searched Lance and skipped the join returns them, ranked and
    plausible, and fails here.

    Every stage's output is checked, not just the last. A stage that emits an out-of-scope
    chunk and a later filter that happens to remove it is a pipeline whose safety depends on
    stage order — and the whole point of a uniform stage is that the order can change.

    Visibility is decided by the store rather than by a list passed in, because the store is
    the thing retrieval will actually consult: ``get_document`` returns ``None`` for a
    document in another workspace or soft-deleted, and the status it reports covers the rest.
    Pass the handle for the workspace the query names.

    Args:
        pipeline: The stages, in the order the runner would call them.
        docstore: The document store scoped to the query's workspace.
        query: The query to run. Its filter carries the scope being enforced.
        expect_results: Require the pipeline to return something. On by default: a pipeline
            that returns nothing satisfies "returned nothing out of scope" without having
            demonstrated anything, so a fixture with no live in-scope chunk is a check that
            cannot fail. Turn it off only where this runs as a runtime assertion, where an
            empty result is an ordinary outcome.

    Returns:
        The final stage's candidates, so a caller can make further assertions.

    Raises:
        AssertionError: A stage emitted a chunk outside the scope, or the pipeline is empty,
            or it returned nothing while ``expect_results`` was set.
    """
    _require(pipeline, "assert_pipeline_enforces_scope needs at least one stage to run")

    candidates: list[Candidate] = []
    for stage in pipeline:
        candidates = await stage.run(query, list(candidates))
        for candidate in candidates:
            document = await docstore.get_document(candidate.chunk.document_id)
            where = (
                f"stage {stage.name!r} returned chunk {candidate.chunk.id!r} of document "
                f"{candidate.chunk.document_id!r}"
            )
            _require(
                document is not None,
                f"{where}, which this store cannot see: it belongs to another workspace or "
                f"has been soft-deleted. The vector table has no column for either, so a "
                f"stage that searched it without hydrating through the document store has "
                f"turned a scoped query into an unscoped one that still looks like it worked",
            )
            _require(
                document is None or document.status is DocumentStatus.INDEXED,
                f"{where}, whose status is "
                f"{document.status.value if document else '?'} rather than 'indexed'. Only "
                f"an indexed document has chunks whose vectors and text are both current; "
                f"the rest are visible to the store and must not be visible to a search",
            )

    if expect_results:
        _require(
            candidates,
            "the pipeline returned no candidates, so this check passed without seeing one. "
            "Give the fixture at least one live, indexed, in-workspace chunk that the query "
            "matches — otherwise a stage that enforces nothing passes too",
        )
    return candidates


# --- configuration -------------------------------------------------------------------------


def assert_local_only_policy_is_enforced(settings: Settings) -> None:
    """Check that a local-only configuration is admitted exactly when nothing leaves.

    ``security.data_policy.cloud_allowed = false`` is a promise about where document content
    goes. It was once kept by consulting the provider's *name*, which meant an ``ollama``
    pointed at another host satisfied it while every prompt and every retrieved passage
    crossed the network — the policy reporting itself satisfied by the configuration it exists
    to forbid.

    Both directions are checked, and the second is not a formality: a predicate that refuses
    an endpoint on ``127.0.0.1`` because the provider is spelled ``openai`` is not a policy,
    it is a ban, and it teaches people to turn the policy off.

    Args:
        settings: A configuration whose data policy is local-only.

    Raises:
        AssertionError: An endpoint that leaves the machine was admitted, an endpoint on this
            machine was refused, or a refusal did not name the endpoint responsible.
    """
    _require(
        not settings.security.data_policy.cloud_allowed,
        "assert_local_only_policy_is_enforced needs a configuration with "
        "security.data_policy.cloud_allowed set to false; there is no policy to enforce "
        "otherwise",
    )

    problems = settings.policy_problems()
    for endpoint in settings.selected_endpoints:
        blamed = [problem for problem in problems if endpoint.describe() in problem]
        if endpoint.leaves_machine:
            _require(
                blamed,
                f"the {endpoint.describe()} is not on this machine, and the configuration "
                f"was admitted anyway. Every prompt and every retrieved passage would cross "
                f"the network under a policy that forbids exactly that, and nothing would "
                f"report it. Problems reported: {problems or 'none'}",
            )
        else:
            _require(
                not blamed,
                f"the {endpoint.describe()} is on this machine and was refused: "
                f"{blamed}. A local-only policy that rejects a genuinely local endpoint is "
                f"not a policy, it is a ban, and the way round it is to switch the policy off",
            )


__all__ = [
    "assert_chunker_contract",
    "assert_connector_contract",
    "assert_embedder_contract",
    "assert_local_only_policy_is_enforced",
    "assert_middleware_contract",
    "assert_parser_contract",
    "assert_pipeline_enforces_scope",
    "assert_protocol_signatures",
    "assert_refuses_oversized_chunks",
    "assert_retrieval_stage_contract",
    "assert_vector_store_is_dimension_agnostic",
    "assert_vector_store_rejects_foreign_vectors",
    "closing",
]
