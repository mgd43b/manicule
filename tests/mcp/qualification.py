"""The bounded research recipe, written as something that runs.

A client with a question and a corpus has to decide four things before it says anything:
which collection it is talking about, how much is in there, what the corpus actually supports,
and — the one that is usually skipped — what it does *not* support. This module is that
procedure, executed over the real MCP protocol against a synthetic corpus, so that the
procedure ``docs/surfaces.md`` §4.2 describes in prose is a thing with a return value rather
than advice.

**It is provider-neutral and there is no model in it.** Every step is a retrieval tool call;
nothing is summarized, ranked or judged by a generator, and :class:`Qualification` is the input
a generator would be given rather than its output. That is deliberate twice over: a hidden
summarizer between search and the client would be an unmeasured component deciding what the
client is allowed to see, and a recipe that needed a hosted model could not run in a test.

**It is bounded on purpose, and the bound is reported rather than assumed.**
:class:`Accounting` counts what each run actually cost — searches, passages asked for, passages
returned, passages left after deduplication, serialized bytes, and estimated generator tokens
where a tokenizer is on the machine. :class:`Budget` is what those are allowed to be. A recipe
whose cost is not measured is a recipe that grows.

**What the corpus below is, and is not.** Three synthetic pages under ``docs.example.test``,
one question they answer and one topic they say nothing about. The ranker is a term-overlap
count — deliberately simple, and *not* manicule's retrieval pipeline. What these fixtures
exercise is the surface around retrieval: that a scope is resolved before it is used, that it
is repeated in every result envelope, that an unknown scope fails closed, and that "the top
five passages held nothing" is assembled into a refusal rather than into silence. How well the
real ranker ranks is :mod:`manicule.evaluation`'s question and is not asked here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from fastmcp import Client

from manicule.app.service import ApplicationService
from manicule.core.anchors import HeadingAnchor
from manicule.core.content import BlockKind, Chunk
from manicule.core.ids import chunk_id
from manicule.core.provenance import LocalSnapshot, Provenance, SourceMetadata
from manicule.core.retrieval import Candidate, Confidence, ConfidenceBand, Context
from manicule.mcp.server import build_server
from manicule.retrieval.retriever import RetrievalResult
from tests.app.fakes import FakeBackend, FakeRetriever, make_document

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from manicule.core.content import Document
    from manicule.core.retrieval import Query

WORKSPACE = "default"

COLLECTION = "Engineering Architecture"
"""The one collection these fixtures create, and the name a client has to resolve."""

BASE_URI = "https://docs.example.test/architecture/"

SUPPORTED_QUESTION = "which component owns admission control"
"""A question the corpus answers, in three places, from three angles."""

ABSENT_TOPIC = "Project Aurora neutrino database ownership"
"""A topic the corpus says nothing about — and the control is not a trick.

``ownership`` is a word this corpus does use, so the control search comes back holding
something rather than empty. That is the case worth testing: an empty result is easy to refuse
correctly, and a plausible one that shares a word with the question is how a confident wrong
answer gets written.
"""

ABSENT_TOPIC_TERMS: tuple[str, ...] = ("aurora", "neutrino")
"""What a passage would have to mention to be about :data:`ABSENT_TOPIC` at all.

Named by the caller rather than inferred, because "is this passage on topic" is the judgment a
generator is for. The recipe only establishes the fact underneath it — that no returned passage
contains any of these — and hands that to whoever writes the sentence.
"""


# --- the corpus -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Page:
    """One synthetic document: where it lives, and the headed passages under it."""

    slug: str
    title: str
    passages: tuple[tuple[str, str], ...]

    @property
    def canonical_uri(self) -> str:
        return f"{BASE_URI}{self.slug}"


PAGES: tuple[Page, ...] = (
    Page(
        slug="control-plane",
        title="Control plane",
        passages=(
            (
                "Admission control",
                "The control plane owns admission control. Every request to place work is "
                "admitted or refused here, and no other component may admit work.",
            ),
            (
                "Ownership",
                "Ownership of admission control sits with the control plane and moves only by "
                "an architecture decision record.",
            ),
            (
                "Placement",
                "Once work is admitted the control plane chooses a runtime to place it on and "
                "records that choice.",
            ),
            (
                "Configuration",
                "Settings are read once at start, and a change needs a restart rather than a "
                "reload.",
            ),
            (
                "Telemetry",
                "Every decision emits a record naming the request, the outcome and the elapsed "
                "time.",
            ),
        ),
    ),
    Page(
        slug="runtime-boundaries",
        title="Runtime boundaries",
        passages=(
            (
                "What a runtime may decide",
                "A runtime decides how to execute the work it is given. It may not admit work "
                "of its own; admission stays with the control plane.",
            ),
            (
                "Isolation",
                "Runtimes share no memory and address each other only through the control plane.",
            ),
            (
                "Backpressure",
                "A runtime that is saturated reports so, and the control plane stops admitting "
                "work to it.",
            ),
            (
                "Naming",
                "A workload carries the same name from submission to completion, so a log line "
                "can be joined to a queue entry.",
            ),
            (
                "Scratch space",
                "Local scratch space is discarded the moment a workload finishes.",
            ),
        ),
    ),
    Page(
        slug="recovery",
        title="Recovery",
        passages=(
            (
                "Replaying the log",
                "Recovery replays the admission log in order, so the set of admitted work is "
                "the same afterward as before.",
            ),
            (
                "Admission during recovery",
                "During recovery the control plane keeps admission control and refuses new "
                "work until the replay finishes.",
            ),
            (
                "Partial failure",
                "A runtime lost mid-replay is reported and its work is re-placed by the "
                "control plane.",
            ),
            (
                "Snapshots",
                "A snapshot is written every hour and names the last entry it contains.",
            ),
            (
                "Verification",
                "After a replay the digest of every entry is compared against the snapshot's.",
            ),
        ),
    ),
)
"""Three pages, five passages each: fifteen chunks, which is what ``collection_counts`` must say.

Two things are arranged here rather than left to chance. The supported question is answered from
three documents rather than one, because a recipe that stops at the first hit and a recipe that
gathers evidence look identical against a corpus where only one document has the answer. And
each page carries passages the evidence queries share no words with, so the whole default
budget — three searches of five — cannot reach the whole collection. That is what makes
"retrieved everything the searches returned" and "read the collection" different claims, which
is the difference every assertion about non-exhaustiveness rests on.
"""

DOCUMENT_COUNT = len(PAGES)
CHUNK_COUNT = sum(len(page.passages) for page in PAGES)


def _document(page: Page) -> Document:
    """One page as an indexed document, carrying the source record a citation quotes."""
    return make_document(
        WORKSPACE,
        source="handbook",
        source_id=f"{page.slug}.html",
        title=page.title,
        provenance=Provenance(
            source=SourceMetadata(
                title=page.title,
                canonical_uri=page.canonical_uri,
                source_id=page.slug,
                version="1",
                section_path=("Engineering", "Architecture"),
            ),
            snapshot=LocalSnapshot(path=f"mirror/{page.slug}.html"),
        ),
    )


def _chunks(document: Document, page: Page) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(
            id=chunk_id(document.id, position, text),
            document_id=document.id,
            text=text,
            embed_text=text,
            anchor=HeadingAnchor(path=(page.title, heading)),
            heading_path=(page.title, heading),
            kind=BlockKind.PROSE,
            position=position,
            token_count=len(text.split()),
        )
        for position, (heading, text) in enumerate(page.passages)
    )


_STOP_WORDS = frozenset(
    {"a", "an", "and", "for", "in", "is", "of", "on", "or", "the", "to", "which", "who"}
)


def _terms(text: str) -> frozenset[str]:
    """The words a match is decided on: lower-cased, punctuation-free, common words dropped."""
    cleaned = "".join(character if character.isalnum() else " " for character in text.lower())
    return frozenset(cleaned.split()) - _STOP_WORDS


@dataclass
class LexicalRetriever(FakeRetriever):
    """Ranks the fixture's chunks by how many of the query's words they contain.

    A subclass of the module's ordinary fake rather than a new component, so that
    :attr:`~tests.app.fakes.FakeRetriever.seen` still records every query that reached a
    retriever — which is how "the unknown collection was refused *before* a search ran" is
    checked as an absence rather than inferred from an error message.

    **It honors the filter's collection scope, and that is the part under test.** A retriever
    that ignored it would let every assertion about scoping pass while the scope did nothing.
    """

    chunks: tuple[Chunk, ...] = ()
    collections_by_document: dict[str, frozenset[str]] = field(
        default_factory=dict[str, frozenset[str]]
    )

    @override
    async def retrieve(self, query: Query) -> RetrievalResult:
        self.seen.append(query)
        wanted = _terms(query.text)
        scope = query.filter.collection_ids
        scored = [
            (overlap, chunk)
            for chunk in self.chunks
            if not scope
            or (self.collections_by_document.get(chunk.document_id, frozenset()) & scope)
            for overlap in (len(_terms(chunk.text) & wanted),)
            if overlap
        ]
        # Highest overlap first, then by chunk id, so two runs of the same query rank the same.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        candidates = [
            Candidate(chunk=chunk, score=overlap / len(wanted))
            for overlap, chunk in scored[: query.limit]
        ]
        # Scored against how much of the *query* was matched, not against the best hit. A
        # ranking normalized to its own winner reports full confidence for one weak match on a
        # question about something the corpus has never heard of, which is exactly the case the
        # control below exists to catch — and it would report it as evidence.
        best = candidates[0].score if candidates else 0.0
        return RetrievalResult(
            context=Context(query=query, passages=tuple(candidates)),
            candidates=candidates,
            confidence=Confidence(
                score=best,
                band=_band(best),
                reason=(
                    f"{len(candidates)} passage(s) share terms with the query"
                    if candidates
                    else "no passage shares a term with the query"
                ),
            ),
        )


def _band(score: float) -> ConfidenceBand:
    """A coarse band, so the control's weak lexical match is visibly weak rather than absent."""
    if score >= 0.6:
        return ConfidenceBand.HIGH
    if score >= 0.3:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW if score > 0 else ConfidenceBand.NONE


async def build_fixture() -> tuple[ApplicationService, FakeBackend]:
    """A service over the ``Engineering Architecture`` fixture, and the backend behind it.

    One collection, so that "``collection_list`` resolves exactly one target" is a claim about
    the fixture rather than about a filter somebody wrote in the test. The backend is handed
    back rather than left to be dug out of the service, because what several of these suites
    assert is that a call left it *unchanged*.
    """
    backend = FakeBackend()
    retriever = LexicalRetriever()
    backend.retriever_ = retriever

    collection = await backend.organization_.create_collection(
        COLLECTION, description="How the control plane, the runtimes and recovery divide work"
    )
    chunks: list[Chunk] = []
    for page in PAGES:
        document = _document(page)
        page_chunks = _chunks(document, page)
        backend.store.add(document, *page_chunks)
        backend.organization_.documents[document.id] = document
        chunks.extend(page_chunks)
        retriever.collections_by_document[document.id] = frozenset({collection.id})
    await backend.organization_.add_to_collection(
        collection.id, [document.id for document in backend.organization_.documents.values()]
    )
    retriever.chunks = tuple(chunks)
    return ApplicationService(backend), backend


# --- what a run of the recipe records ----------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What one qualification is allowed to cost.

    Declared rather than discovered: a ceiling written down before the run is a decision, and a
    ceiling read off the run afterward is a description of whatever happened.
    """

    searches: int = 4
    """Three for evidence and one for the control. A fourth evidence search needs a named gap."""

    limit: int = 5
    """Passages per search. Small enough that a second search is cheaper than a wider first."""

    payload_bytes: int = 60_000
    """Total serialized MCP results. This is what actually reaches a client's context."""

    generator_tokens: int = 4_000
    """Estimated tokens the assembled evidence would contribute, where a tokenizer exists."""


@dataclass(frozen=True)
class Accounting:
    """What one qualification actually cost."""

    searches: int
    requested_passages: int
    """The sum of every ``limit`` asked for."""

    returned_passages: int
    """How many came back. Below ``requested_passages`` whenever the corpus ran out first."""

    deduplicated_passages: int
    """How many survive deduplication by document and heading — what synthesis would read."""

    payload_bytes: int
    """Bytes of serialized MCP result across every call, ``tools/list`` included."""

    generator_tokens: int | None
    """Estimated generator tokens for the deduplicated evidence, or ``None``.

    ``None`` means this machine has no BPE vocabulary seeded, not that the answer is zero — see
    :class:`~manicule.retrieval.tokens.ContextTokenCounter`, which refuses rather than
    downloading one. Reported as absent so a run on an unseeded host is visibly unmeasured.
    """

    def within(self, budget: Budget) -> bool:
        """Whether every declared ceiling held."""
        return (
            self.searches <= budget.searches
            and self.payload_bytes <= budget.payload_bytes
            and (self.generator_tokens is None or self.generator_tokens <= budget.generator_tokens)
        )


@dataclass(frozen=True)
class Passage:
    """One retrieved passage, reduced to what a citation needs and what synthesis reads."""

    document_id: str
    title: str
    canonical_uri: str | None
    heading_path: tuple[str, ...]
    text: str

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """What two hits have to share to be the same passage twice."""
        return (self.document_id, self.heading_path)


@dataclass(frozen=True)
class Finding:
    """One search: what was asked, under what scope, and what came back."""

    query: str
    scope: tuple[str, ...]
    requested: int
    passages: tuple[Passage, ...]
    confidence_band: str


@dataclass(frozen=True)
class Qualification:
    """Everything one run established, and what it cost to establish it.

    The five things it keeps apart are kept apart on purpose. Collection identity and computed
    counts are facts about the corpus. Supported evidence is what was found. The control is what
    was looked for and not found. Provenance completeness is whether each passage can be cited
    at all. And the counts are what says how much of the corpus was never looked at — because
    top-`limit` retrieval is a sample, and a sample that found nothing is not a corpus that
    holds nothing.
    """

    collection_name: str
    collection_id: str
    collections_seen: int
    documents: int
    chunks: int
    supported: tuple[Finding, ...]
    control: Finding
    evidence: tuple[Passage, ...]
    accounting: Accounting

    @property
    def control_support(self) -> tuple[Passage, ...]:
        """Passages from the control search that mention the topic at all. Expected empty."""
        return tuple(
            passage
            for passage in self.control.passages
            if any(term in passage.text.lower() for term in ABSENT_TOPIC_TERMS)
        )

    @property
    def citable(self) -> tuple[Passage, ...]:
        """Evidence carrying everything a claim-level citation needs."""
        return tuple(
            passage
            for passage in self.evidence
            if passage.canonical_uri and passage.title and passage.heading_path
        )

    def refusal(self) -> str:
        """The sentence the control's result licenses — assembled here, not generated.

        Says what was searched and how much was not looked at, because "no supported claim" and
        "there is no such thing" are different statements and only the first is true.
        """
        return (
            f"Nothing in {self.collection_name} supports a claim about {ABSENT_TOPIC}. "
            f"The collection holds {self.documents} document(s) and {self.chunks} chunk(s); "
            f"the search returned {len(self.control.passages)} passage(s) and none of them "
            f"mentions {' or '.join(ABSENT_TOPIC_TERMS)}. Top-{self.control.requested} "
            f"retrieval samples a ranking, so this is the absence of support rather than "
            f"proof of absence."
        )


# --- the recipe -------------------------------------------------------------------------------

EVIDENCE_QUERIES: tuple[str, ...] = (
    SUPPORTED_QUESTION,
    "what may a runtime decide for itself",
    "admission control during recovery",
)
"""Three angles on one question, which is the whole of the default evidence budget.

Not three rephrasings: each one is aimed at a different document, so a fourth search has to be
justified by a gap none of these covered rather than by the first one having been vague.
"""


class _Transcript:
    """One MCP session, with every result weighed on the way past."""

    def __init__(self, client: Client[Any]) -> None:
        self._client = client
        self.bytes = 0

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one tool and return its envelope, counting what it serialized to.

        The bytes are measured on the envelope rather than estimated from the passages,
        because the envelope is what crosses the wire and a client pays for all of it —
        scores, anchors and provenance included.
        """
        result = await self._client.call_tool(tool, arguments)
        envelope: dict[str, Any] = dict(result.structured_content or {})
        self.bytes += len(json.dumps(envelope, sort_keys=True).encode("utf-8"))
        return envelope


def _passages(envelope: dict[str, Any]) -> tuple[Passage, ...]:
    data: dict[str, Any] = envelope["data"]
    hits: Sequence[dict[str, Any]] = data["hits"]
    kept: list[Passage] = []
    for hit in hits:
        record: dict[str, Any] = hit.get("provenance") or {}
        canonical: object = record.get("canonical_uri")
        kept.append(
            Passage(
                document_id=str(hit["document_id"]),
                title=str(hit["title"]),
                canonical_uri=str(canonical) if canonical is not None else None,
                heading_path=tuple(str(part) for part in hit["heading_path"]),
                text=str(hit["text"]),
            )
        )
    return tuple(kept)


def _finding(query: str, requested: int, envelope: dict[str, Any]) -> Finding:
    data: dict[str, Any] = envelope["data"]
    return Finding(
        query=query,
        scope=tuple(str(name) for name in data["collections"]),
        requested=requested,
        passages=_passages(envelope),
        confidence_band=str(data["confidence_band"] or ""),
    )


def deduplicate(findings: Iterable[Finding]) -> tuple[Passage, ...]:
    """Every distinct passage across several searches, in the order first seen.

    Three searches aimed at one question return the same passage more than once by design —
    that is what agreement between angles looks like — and paying for it three times is not.
    """
    seen: set[tuple[str, tuple[str, ...]]] = set()
    kept: list[Passage] = []
    for finding in findings:
        for passage in finding.passages:
            if passage.key not in seen:
                seen.add(passage.key)
                kept.append(passage)
    return tuple(kept)


def estimate_tokens(passages: Iterable[Passage]) -> int | None:
    """What the evidence would cost a generator, or ``None`` where nothing can count it."""
    from manicule.retrieval.tokens import ContextTokenCounter  # noqa: PLC0415 - optional on a host

    try:
        counter = ContextTokenCounter()
    except Exception:  # noqa: BLE001 - any refusal to load a vocabulary means "not measured"
        return None
    return sum(counter.count(passage.text) for passage in passages)


async def qualify(service: ApplicationService, *, budget: Budget | None = None) -> Qualification:
    """Run the recipe end to end over the MCP protocol, and report what it established.

    The order is the order the server's own instructions give a client, and it is not
    interchangeable: the collection is resolved before anything is scoped to it, counts come
    from the operation that computes them, and every search names its scope again because
    nothing between calls remembers it.
    """
    limit = (budget or Budget()).limit
    async with Client(build_server(service)) as client:
        transcript = _Transcript(client)

        listed = await transcript.call("collection_list", {})
        collections: Sequence[dict[str, Any]] = listed["data"]["collections"]
        target = next(item for item in collections if item["name"] == COLLECTION)

        counted = await transcript.call("collection_counts", {"collection_id": target["id"]})

        gathered: list[Finding] = []
        for query in EVIDENCE_QUERIES:
            envelope = await transcript.call(
                "search", {"query": query, "collections": [COLLECTION], "limit": limit}
            )
            gathered.append(_finding(query, limit, envelope))
        supported = tuple(gathered)

        control = _finding(
            ABSENT_TOPIC,
            limit,
            await transcript.call(
                "search", {"query": ABSENT_TOPIC, "collections": [COLLECTION], "limit": limit}
            ),
        )

    evidence = deduplicate(supported)
    searches = len(supported) + 1
    returned = sum(len(finding.passages) for finding in (*supported, control))
    return Qualification(
        collection_name=str(target["name"]),
        collection_id=str(target["id"]),
        collections_seen=len(collections),
        documents=int(counted["data"]["documents"]),
        chunks=int(counted["data"]["chunks"]),
        supported=supported,
        control=control,
        evidence=evidence,
        accounting=Accounting(
            searches=searches,
            requested_passages=searches * limit,
            returned_passages=returned,
            deduplicated_passages=len(evidence),
            payload_bytes=transcript.bytes,
            generator_tokens=estimate_tokens(evidence),
        ),
    )
