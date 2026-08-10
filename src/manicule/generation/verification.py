"""Citation verification: the three-level ladder, and what it costs.

The ladder (``docs/generation.md`` §3.3):

===== ============================================================ ======================
Level Check                                                        Catches
===== ============================================================ ======================
0     the slot is an integer in ``1..len(context.passages)``        invention
1     the passage's anchor is not ``Unlocated``                     a location the parser
                                                                    itself disclaims
2     ``Parser.resolve`` over the retained bytes returns text       anchor drift, missing
      containing what the chunk claims                              bytes, stale parsers
===== ============================================================ ======================

Level 2 is the one that needs justifying, because ``CONTRIBUTING.md`` already imposes a
round-trip obligation on every parser and ``assert_parser_contract`` already enforces it. It
is still not sufficient, for four reasons that have nothing to do with parser quality: that
suite runs against **fixtures**, not against this corpus; the bytes can be **gone**, because
a blob store can lose a file and retention is configurable off; a stored conversation
**replays** its citations after the document may have been re-ingested, which is the case
where this is the only check there is; and a **restore or repair** can leave anchors written
by code that no longer runs.

**Verification starts before the first token**, because it depends only on
:class:`~manicule.core.retrieval.Context`, which is already known. By the time a marker
appears in the stream the answer for that passage is usually already in hand — which is what
makes per-citation verification affordable on the answer path rather than a thing that gets
skipped for latency.

**The predicate is not defined here.** It is
:func:`manicule.testing.normalise.contains_claimed_text`, the same call the parser
conformance suite makes, because two notions of "this anchor resolves" would drift and the
drift would show up as citations that pass CI and fail in production.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from manicule.core.anchors import Anchor, Unlocated
from manicule.core.content import Document, RawDocument
from manicule.core.lifecycle import HealthReport
from manicule.core.protocols import Parser
from manicule.core.retrieval import Candidate, Context
from manicule.generation.answers import CitationDrop, DropReason, Verification
from manicule.testing.normalise import contains_claimed_text


@runtime_checkable
class DocumentLookup(Protocol):
    """The one read a citation needs of the document store."""

    async def get_document(self, document_id: str) -> Document | None: ...


async def load_documents(lookup: DocumentLookup, context: Context) -> dict[str, Document]:
    """Every document a context's passages point into, read once each.

    Done up front rather than lazily, because two things need it and neither can wait: the
    prompt's slot labels carry the document's title, and a citation carries its ``uri`` and
    ``title`` — neither of which is on a :class:`~manicule.core.content.Chunk`. Reading them
    per citation would be the same rows fetched twice, once too late.

    A document absent from the result is one that is no longer in the index. Its passages are
    still shown to the model — they are what retrieval found — and citations to them are
    dropped, because a citation that cannot name the document it points into is not one.
    """
    found: dict[str, Document] = {}
    for document_id in dict.fromkeys(p.chunk.document_id for p in context.passages):
        document = await lookup.get_document(document_id)
        if document is not None:
            found[document_id] = document
    return found


@runtime_checkable
class BlobReader(Protocol):
    """The one read verification needs of the blob store."""

    async def get(self, digest: str) -> bytes | None: ...


@runtime_checkable
class ParserChainLike(Protocol):
    """The part of a configured parser chain :class:`ChainRouter` reads.

    Structural, so ``manicule.parsers.chain.ParserChain`` satisfies it without this module
    importing a single parser library — which is what keeps the generation package free of
    pdfium, tree-sitter and the rest.
    """

    @property
    def parsers(self) -> Mapping[str, Parser]: ...

    def resolve(self, media_type: str) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class OpenSource:
    """A document's retained bytes with the parser that can read them.

    Obtained once per *document* and reused across every anchor into it, which is where the
    saving is: with ``final_top_k`` between 3 and 10 and passages frequently sharing a
    document, an ordinary answer performs one to three blob reads. Note what is **not**
    shared: :meth:`~manicule.core.protocols.Parser.resolve` is an independent call per
    anchor, so the parse itself is per anchor and not per document.
    """

    parser: Parser
    raw: RawDocument

    async def resolve(self, anchor: Anchor) -> str | None:
        return await self.parser.resolve(anchor, self.raw)


@runtime_checkable
class AnchorResolver(Protocol):
    """Where level 2 gets its evidence.

    Deliberately **not** the thing that looks a document up. Every citation needs its
    document — a citation that cannot name the document it points into is not a citation —
    and that read has to happen whether or not level 2 is reachable. Folding it in here would
    make a configuration with retention switched off unable to produce citations at all,
    which is the opposite of the graceful degradation §3.7 asks for.
    """

    @property
    def can_resolve(self) -> bool:
        """Whether level 2 is reachable at all in this configuration."""
        ...

    async def open(self, document: Document) -> OpenSource | None:
        """The bytes and parser for ``document``, or ``None`` if either is unavailable."""
        ...


@dataclass(frozen=True, slots=True)
class ChainRouter:
    """Picks the parser that wrote an anchor, given a configured chain.

    **Prefers the parser the document was actually parsed with**, recorded as
    ``parser_used`` in the document's metadata by the ingest chain, and falls back to the
    head of the current chain for its media type. Trying every parser in the chain until one
    returned text would be worse than useless: a parser that never saw this document could
    return text containing the chunk's own words by coincidence, and the citation would then
    be certified by the wrong reader.
    """

    chain: ParserChainLike

    def parser_for(self, document: Document) -> Parser | None:
        used = document.metadata.get("parser_used")
        if isinstance(used, str) and used in self.chain.parsers:
            return self.chain.parsers[used]
        for name in self.chain.resolve(document.media_type):
            parser = self.chain.parsers.get(name)
            if parser is not None:
                return parser
        return None


@dataclass(frozen=True, slots=True)
class RetainedBytesResolver:
    """Resolves anchors over the bytes ingest kept, never by re-fetching."""

    blobs: BlobReader
    router: ChainRouter

    @property
    def can_resolve(self) -> bool:
        return True

    async def open(self, document: Document) -> OpenSource | None:
        if document.original_ref is None:
            return None
        data = await self.blobs.get(document.original_ref)
        if data is None:
            return None
        parser = self.router.parser_for(document)
        if parser is None:
            return None
        return OpenSource(
            parser,
            RawDocument(
                source_id=document.source_id,
                uri=document.uri,
                media_type=document.media_type,
                content=data,
                metadata=dict(document.metadata),
            ),
        )


@dataclass(frozen=True, slots=True)
class UnverifiableSource:
    """The resolver for a configuration that cannot reach level 2.

    Source bytes are not retained, or nothing wired a parser and a blob store. Level 2 is
    then *impossible* rather than failing, so the ceiling drops to level 1 and the answer
    **names the level it reached** — reporting the same word for two different amounts of
    checking is the failure this distinction exists to prevent.
    """

    reason: str

    @property
    def can_resolve(self) -> bool:
        return False

    async def open(self, document: Document) -> OpenSource | None:
        del document
        return None


@dataclass(slots=True)
class _Cache:
    """Verified levels, keyed by ``(chunk_id, version_token)``.

    Chunk ids are content-derived and ``version_token`` changes whenever the document does,
    so the key is exact and can never go stale — the same property the token-count cache
    relies on. A verified anchor stays verified until its document changes, which across a
    corpus's life makes this close to a once-per-chunk cost rather than a per-query one.
    """

    limit: int = 10_000
    _entries: OrderedDict[tuple[str, str], Verification] = field(
        default_factory=OrderedDict[tuple[str, str], Verification]
    )
    hits: int = 0

    def get(self, key: tuple[str, str]) -> Verification | None:
        found = self._entries.get(key)
        if found is None:
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return found

    def put(self, key: tuple[str, str], level: Verification) -> None:
        if self.limit <= 0:
            return
        self._entries[key] = level
        self._entries.move_to_end(key)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)


@dataclass(frozen=True, slots=True)
class SlotVerdict:
    """What verification concluded about one slot.

    Carries the :class:`~manicule.core.content.Document` because that is where a citation's
    ``uri`` and ``title`` come from — neither is on a :class:`~manicule.core.content.Chunk`,
    and re-reading the document at binding time would be a second round trip for a row this
    already had in hand.
    """

    reached: Verification
    drop: CitationDrop | None = None
    document: Document | None = None

    @property
    def survives(self) -> bool:
        return self.drop is None and self.document is not None


class CitationVerifier:
    """Verifies the passages of a context, concurrently with the model call.

    Construct one per process and call :meth:`start` per answer. The run it returns has
    already begun working by the time it is handed back, so on a warm cache it has finished
    before the request reaches the provider.
    """

    def __init__(
        self,
        resolver: AnchorResolver,
        *,
        timeout_s: float = 5.0,
        cache_entries: int = 10_000,
    ) -> None:
        self._resolver = resolver
        self._timeout_s = timeout_s
        self._cache = _Cache(limit=cache_entries)

    @property
    def ceiling(self) -> Verification:
        """The strongest level this configuration can reach."""
        return Verification.RESOLVED if self._resolver.can_resolve else Verification.LOCATED

    @property
    def cache_hits(self) -> int:
        return self._cache.hits

    async def health(self) -> HealthReport:
        """Degraded when level 2 is unreachable, with the setting named.

        Knowable at startup rather than per query, because it is a property of configuration —
        so it is a health state with a remedy rather than a surprise attached to an answer.
        """
        if self._resolver.can_resolve:
            return HealthReport.healthy("citations verify against retained source bytes")
        return HealthReport.degraded(
            "citations can only be verified to 'located': there are no retained source bytes "
            "to resolve an anchor against, so anchor drift and a lost blob are both invisible",
            remedy="set storage.retain_source_bytes = true and re-sync, or accept that answers "
            "will report verification_level 'located' rather than 'resolved'",
        )

    def start(
        self,
        context: Context,
        documents: Mapping[str, Document],
        *,
        started_at: float | None = None,
    ) -> VerificationRun:
        """Begin verifying every passage of ``context``. Returns immediately."""
        began = started_at if started_at is not None else time.monotonic()
        return VerificationRun(
            passages=context.passages,
            documents=documents,
            resolver=self._resolver,
            cache=self._cache,
            ceiling=self.ceiling,
            deadline=began + self._timeout_s,
        )


class VerificationRun:
    """One answer's worth of verification, in flight.

    Work is grouped by document and each group runs as its own task, so a slow blob does not
    delay a citation into another document. :meth:`aclose` must run on every path — an
    abandoned run holds tasks still reading blobs for an answer nobody will see.
    """

    def __init__(
        self,
        *,
        passages: Sequence[Candidate],
        documents: Mapping[str, Document],
        resolver: AnchorResolver,
        cache: _Cache,
        ceiling: Verification,
        deadline: float,
    ) -> None:
        self._passages = tuple(passages)
        self._documents = dict(documents)
        self._resolver = resolver
        self._cache = cache
        self._ceiling = ceiling
        self._deadline = deadline
        self._verdicts: dict[int, SlotVerdict] = {}
        self._ready: dict[int, asyncio.Event] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._start()

    @property
    def slots_offered(self) -> int:
        return len(self._passages)

    @property
    def ceiling(self) -> Verification:
        return self._ceiling

    def _start(self) -> None:
        by_document: dict[str, list[int]] = {}
        for index, passage in enumerate(self._passages):
            slot = index + 1
            self._ready[slot] = asyncio.Event()
            anchor = passage.chunk.anchor
            if isinstance(anchor, Unlocated):
                self._settle(
                    slot,
                    SlotVerdict(
                        Verification.BOUND,
                        CitationDrop(
                            slot=slot,
                            reason=DropReason.UNLOCATED,
                            reached=Verification.BOUND,
                            detail=anchor.reason or "the parser could not determine a location",
                        ),
                    ),
                )
            else:
                by_document.setdefault(passage.chunk.document_id, []).append(slot)

        for document_id, slots in by_document.items():
            self._tasks.append(asyncio.create_task(self._verify_document(document_id, slots)))

    def _settle(self, slot: int, verdict: SlotVerdict) -> None:
        self._verdicts[slot] = verdict
        self._ready[slot].set()

    def _settle_remaining(
        self, slots: Sequence[int], detail: str, document: Document | None = None
    ) -> None:
        """Give every slot still waiting a verdict.

        Without this a cancelled or failed group leaves a slot waiting for an event nobody
        will set, so the awaiting side sits out its whole budget and reports a timeout for a
        defect that is not one.
        """
        for slot in slots:
            if slot not in self._verdicts:
                self._settle(
                    slot,
                    SlotVerdict(
                        Verification.LOCATED,
                        CitationDrop(
                            slot=slot,
                            reason=DropReason.UNRESOLVABLE,
                            reached=Verification.LOCATED,
                            detail=detail,
                        ),
                        document,
                    ),
                )

    async def _verify_document(self, document_id: str, slots: Sequence[int]) -> None:
        """Settle every slot pointing into one document, sharing its reads across them."""
        try:
            document = self._documents.get(document_id)
            if document is None:
                self._settle_remaining(
                    slots,
                    f"document {document_id} is no longer in the index, so this citation "
                    f"cannot name what it points into",
                )
                return
            await self._verify_slots(document, slots)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a parser's own bug drops citations, not answers
            self._settle_remaining(slots, f"{type(exc).__name__}: {exc}")
        finally:
            self._settle_remaining(slots, "verification did not complete")

    async def _verify_slots(self, document: Document, slots: Sequence[int]) -> None:
        if self._ceiling is Verification.LOCATED:
            # Level 2 is impossible in this configuration rather than failing, so every
            # located anchor has already reached the strongest level there is.
            for slot in slots:
                self._settle(slot, SlotVerdict(Verification.LOCATED, None, document))
            return
        version = document.version_token or document.content_hash
        pending = [slot for slot in slots if not self._settle_from_cache(slot, version, document)]
        if not pending:
            return
        source = await self._resolver.open(document)
        if source is None:
            self._settle_remaining(
                pending,
                f"the retained bytes for document {document.id} are unavailable, so the anchor "
                f"cannot be resolved",
                document,
            )
            return
        for slot in pending:
            chunk = self._passages[slot - 1].chunk
            resolved = await source.resolve(chunk.anchor)
            reached = (
                Verification.RESOLVED
                if contains_claimed_text(resolved, chunk.text)
                else Verification.LOCATED
            )
            self._cache.put((chunk.id, version), reached)
            detail = (
                "the anchor resolves to text that does not contain what this chunk claims"
                if resolved is not None
                else "the anchor resolves to nothing over the retained bytes"
            )
            self._settle(slot, self._verdict_for(slot, reached, detail, document))

    def _settle_from_cache(self, slot: int, version: str, document: Document) -> bool:
        cached = self._cache.get((self._passages[slot - 1].chunk.id, version))
        if cached is None:
            return False
        self._settle(
            slot,
            self._verdict_for(
                slot, cached, "a previous run could not resolve this anchor", document
            ),
        )
        return True

    def _verdict_for(
        self, slot: int, reached: Verification, detail: str, document: Document
    ) -> SlotVerdict:
        if reached.at_least(self._ceiling):
            return SlotVerdict(reached, None, document)
        return SlotVerdict(
            reached,
            CitationDrop(slot=slot, reason=DropReason.UNRESOLVABLE, reached=reached, detail=detail),
            document,
        )

    async def verdict(self, slot: int) -> SlotVerdict:
        """What verification concluded about ``slot``, waiting out the remaining budget.

        Level 0 is answered here and costs nothing: a slot outside ``1..N`` never had a
        passage, so there is nothing to wait for.

        A slot whose verification has not finished when its marker must be emitted waits out
        the remainder of the budget and is then **dropped**, with
        :attr:`~manicule.generation.answers.DropReason.VERIFICATION_TIMEOUT` — distinct from
        ``unresolvable``, because one means the disk is slow and the other means the citation
        is wrong. Dropping a possibly-good citation is the uncomfortable half of that trade;
        sending an unverified one, under a design whose entire claim is verification, is the
        unacceptable half.
        """
        if not 1 <= slot <= len(self._passages):
            return SlotVerdict(
                Verification.BOUND,
                CitationDrop(
                    slot=slot,
                    reason=DropReason.OUT_OF_RANGE,
                    detail=f"slot {slot} named, but only {len(self._passages)} were offered",
                ),
            )
        settled = self._verdicts.get(slot)
        if settled is not None:
            return settled
        remaining = max(self._deadline - time.monotonic(), 0.0)
        try:
            await asyncio.wait_for(self._ready[slot].wait(), remaining)
        except TimeoutError:
            return SlotVerdict(
                Verification.LOCATED,
                CitationDrop(
                    slot=slot,
                    reason=DropReason.VERIFICATION_TIMEOUT,
                    reached=Verification.LOCATED,
                    detail="verification did not finish inside llm.citation_verify_timeout_s",
                ),
            )
        return self._verdicts[slot]

    async def aclose(self) -> None:
        """Cancel any verification still running, and wait for it to stop."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:  # noqa: SIM105 - suppress() would also swallow a cancellation of *this* task
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()


__all__ = [
    "AnchorResolver",
    "BlobReader",
    "ChainRouter",
    "CitationVerifier",
    "DocumentLookup",
    "OpenSource",
    "ParserChainLike",
    "RetainedBytesResolver",
    "SlotVerdict",
    "UnverifiableSource",
    "VerificationRun",
    "load_documents",
]
