"""``[[wikilinks]]`` as typed chunk relations.

A plugin rather than a core parser rule, and that is the first decision worth reading.
``[[wikilink]]`` is a memory-and-Obsidian idiom, not a markdown one: CommonMark has no such
syntax, and a corpus of ordinary documentation that happened to contain double brackets would
acquire a graph nobody asked for. Keeping the convention out of the core parser is right on its
own, and it exercises the public extension path on a component that does real work rather than
only on the example.

**What it does.** One middleware, registered under ``keys.MIDDLEWARE``. After a document is
stored it scans that document's chunk text for links, resolves each to a document, and writes a
:class:`~manicule.core.organization.ChunkEdge` — then looks the other way, for documents already
stored that link to *this* one, and writes those. The second half is what makes a forward
reference work: a link may name a memory that does not exist yet, that is a deliberate part of
the writing convention, and an unresolved link is therefore normal rather than an error.

**Why the second half is a search rather than a table of pending links.** ``chunk_relations`` has
real foreign keys to ``chunks``, so an edge to a document that does not exist cannot be stored —
there is nothing for the row to point at. The two designs available were a table of unresolved
links of its own, or re-resolving on each target publish; this is the second. It needs no new
schema, it cannot drift out of step with the edges it would be describing, and what it costs is
one bounded lexical query per stored document.

**The rules are versioned and the edges are not versioned by them.** What decides a link, how
two spellings of a target are matched, and where the line runs between a declared link and a
passing mention all live in :mod:`manicule_plugin_wikilinks.links`, and this module digests that
file into :class:`~manicule.core.fingerprints.RelationRules`. The pipeline records the result per
document, so a corrected rule makes the corpus visibly stale and the existing documents are
selected by the repair that already exists — rather than the graph quietly keeping edges the old
rules produced while every other fingerprint reports itself current.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from pydantic import BaseModel, Field

from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.core.fingerprints import RelationRules
from manicule.core.organization import ChunkRelationType
from manicule.core.protocols import ChunkRelationExtractor, Middleware
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule_plugin_wikilinks.links import Shape, links_in, normalize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from manicule.core.content import Chunk, Document
    from manicule.core.retrieval import Candidate, Filter

EXTRACTOR = "wikilink"
"""The name this extractor records in every document's relation lineage.

A name rather than only a digest, on :data:`manicule.ingest.glossary_lineage.DETECTOR`'s
reasoning: a digest tells two versions of one extractor apart and cannot tell a typo fix from a
different extraction strategy, and a reader of the stored column has to be able to see which
kind of change they are looking at.
"""

SOURCES: tuple[tuple[str, str], ...] = (
    ("manicule_plugin_wikilinks", "links.py"),
    ("manicule_plugin_wikilinks", "__init__.py"),
)
"""Every file whose bytes decide which edges a document contributes, as package and file name.

``links.py``
    The rules: the link pattern, the alias and fragment stripping, the normalization that
    decides whether two spellings are one target, and the shape test that types an edge.

``__init__.py``
    This file, because the resolution policy is here rather than there — which chunk of a target
    an edge points at, whether a self-link is written, and how far the inbound search looks. Each
    of those decides a stored row just as surely as a regular expression does.

Read as bytes with CRLF folded to LF, so two checkouts of one commit agree and a corpus restored
onto a differently-configured machine is not stale for a reason nobody can act on. The digest is
deliberately too sensitive rather than too lenient: a comment-only edit moves it and makes the
corpus stale, which costs a re-scan, while a normalization that skipped comments would one day
drop a real change silently.
"""

MIN_SEARCHABLE_SLUG_TOKENS = 2
"""Fewest tokens a slug must have before the inbound search is worth running.

A one-token slug is a word, and searching a corpus for a word returns the corpus. The documents
that link to such a target still acquire their edges when they are themselves scanned; what is
skipped is the acceleration, not the result.
"""


class WikilinkConfig(BaseModel):
    """Configuration for :class:`WikilinkMiddleware`.

    Set under ``plugins.config."middleware.wikilinks"``. Anything not declared here is rejected,
    so a typo fails at startup rather than doing nothing quietly.
    """

    inbound_limit: int = Field(
        default=200,
        ge=0,
        description="How many stored chunks the inbound search may consider when a document is "
        "published. It bounds the cost of resolving forward references and nothing else: a "
        "document whose linkers fall outside it still gets its edges the next time those "
        "documents are scanned or repaired. Zero switches the inbound pass off entirely, which "
        "leaves forward references to the repair.",
    )
    sources: tuple[str, ...] = Field(
        default=(),
        description="Connector instance names whose documents may be link targets. Empty means "
        "every source in the workspace, which is right for an installation whose corpus is one "
        "directory of memories and wrong for one that also indexes a wiki: a slug is a filename "
        "stem, stems are not unique across sources, and a link resolving into a corpus nobody "
        "was writing about is a wrong edge rather than a missing one.",
    )


@runtime_checkable
class RelationCorpus(Protocol):
    """The narrow store surface this middleware needs, declared rather than assumed.

    Four methods and one attribute out of the six protocols
    :class:`~manicule.storage.docstore.SqliteDocStore` satisfies. Written out here because no
    single shipped protocol carries them all — ``document_chunks`` and ``search_lexical`` belong
    to the document store, ``relate`` to :class:`~manicule.core.protocols.ChunkRelationStore` —
    and a plugin that cast its way to the union would be claiming a shape nothing checks. This one
    is checked, once, at construction, and the refusal names what is missing.

    ``workspace_id`` is here because :class:`~manicule.core.retrieval.Filter` requires it and has
    no default, deliberately: it is the one field carrying a tenancy boundary, and a boundary that
    can be forgotten is not one. The handle already has the answer, so this reads it rather than
    letting configuration supply one that could disagree with the handle it is filtering.
    """

    @property
    def workspace_id(self) -> str: ...

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]: ...

    async def list_documents(
        self,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Document]: ...

    async def search_lexical(
        self,
        text: str,
        k: int,
        filter: Filter | None = None,  # noqa: A002 - mirrors the protocol
    ) -> list[Candidate]: ...

    async def relate(
        self, source_chunk_id: str, target_chunk_id: str, relation_type: ChunkRelationType
    ) -> None: ...


_RELATION_OF = {Shape.DECLARED: ChunkRelationType.LINKS_TO, Shape.PROSE: ChunkRelationType.MENTIONS}
"""How a link's written shape becomes a stored relation type. The whole of the mapping."""

_INDEX_PAGE = 500
"""Documents per page while building the slug index. A page size, not a limit."""


@cache
def rules_digest() -> str:
    """A digest over :data:`SOURCES`, which is this extractor's identity.

    Cached, as :func:`manicule.ingest.glossary_lineage.rules_digest` is, because the files cannot
    change under a running process and the pipeline asks for this once per construction — while
    a suite driving many pipelines would otherwise re-read and re-hash them for every one.

    Derived rather than maintained by hand, for the reason
    :mod:`manicule.ingest.glossary_lineage` gives about the stage upstream: a version number
    somebody has to remember to move is a number that will one day not be moved, and the failure
    is silent — a corpus that reports itself current while serving edges from rules that have
    been corrected.
    """
    digest = hashlib.sha256()
    for package, name in SOURCES:
        digest.update(files(package).joinpath(name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def slug_of(document: Document) -> str:
    """The normalized slug a link has to name to reach ``document``.

    Taken from the **last segment of** ``source_id``, which for a filesystem source is the file's
    path and therefore its stem — the same string the corpus convention puts in ``name:`` at the
    top of each file. Not from the title: a title is display text that changes, and resolving
    through it would make renaming a document break every link into it, which is the failure
    ``document_create`` picks a slug rather than a title to avoid.

    Both separators are split on rather than using :class:`~pathlib.PurePath`, whose idea of a
    separator depends on the machine running this: a corpus written on one and read on another
    must resolve identically.
    """
    last = document.source_id.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return normalize(last)


def _entry_of(chunks: Sequence[Chunk]) -> str:
    """The chunk an edge into a document points at: the lowest-positioned one.

    By ``position`` rather than by the order a store happened to return them, and in one function
    because both directions ask the question. Reading the first element in one place and the
    minimum in the other would make an inbound edge and an outbound edge to the same document
    point at different chunks on any store whose ordering is not what the caller assumed.
    """
    return min(chunks, key=lambda chunk: chunk.position).id


class WikilinkMiddleware(Middleware):
    """Writes chunk relations for the links a document contains, and for links into it.

    Inherits :class:`~manicule.core.protocols.Middleware` for the pass-through defaults: the only
    hook it overrides is ``after_store``, because an edge names stored chunks and there are none
    to name until the document is committed. It touches no text at all — ``mutates_embedded_text``
    stays false and the runner's immutability digests are never in question.
    """

    name = "wikilinks"

    def __init__(self, store: RelationCorpus, config: WikilinkConfig) -> None:
        self._store = store
        self._config = config
        self._slugs: dict[str, str] | None = None
        """Normalized slug to document id, or ``None`` before the first scan.

        Built once per pipeline and kept current by adding each document as it is stored. An
        outgoing link that misses it is a forward reference — normal, not an error — and is
        resolved from the other end when its target is published.
        """

    def relation_rules(self) -> RelationRules:
        """What decides the edges this writes. Computed once; the files cannot change under it."""
        return RelationRules(extractor=EXTRACTOR, rules=rules_digest())

    @override
    async def after_store(self, document: Document) -> None:
        """Write this document's outgoing edges, then the ones pointing at it.

        **Both directions, every time**, and the second is the part that is easy to leave out. A
        corpus is written over months and links forward constantly; resolving only outgoing links
        would leave every forward reference permanently unresolved, because the document
        containing it is never re-ingested by the arrival of its target.

        Failure is not caught here. The pipeline treats a raising ``after_store`` as this hook's
        problem rather than the document's — the document stays published — and, crucially, does
        not stamp the relation lineage, so the document remains selected by the next repair.
        Swallowing an error here would trade that for a document stamped as scanned by an
        extractor that did not finish.
        """
        chunks = await self._store.document_chunks(document.id)
        if not chunks:
            return
        slugs = await self._index()
        if self._resolvable(document):
            await self._claim(slugs, slug_of(document), document.id)
        await self._write_outbound(document, chunks, slugs)
        await self._write_inbound(document, chunks)

    # --- the two directions ----------------------------------------------------------------

    async def _write_outbound(
        self, document: Document, chunks: Sequence[Chunk], slugs: Mapping[str, str]
    ) -> None:
        """One edge per link this document contains that resolves to a document that exists.

        The index is passed in rather than fetched again, so this and the registration above are
        reading the same mapping — one that already has this document in it.
        """
        first: dict[str, str | None] = {}
        for chunk in chunks:
            for link in links_in(chunk.text):
                target = slugs.get(link.target)
                if target is None or target == document.id:
                    # Unresolved, or a document linking to itself. The first is a forward
                    # reference and the second is an edge that says nothing; neither is a
                    # failure and neither is worth a row.
                    continue
                if target not in first:
                    first[target] = await self._entry_chunk(target)
                entry = first[target]
                if entry is not None:
                    await self._relate(chunk.id, entry, _RELATION_OF[link.shape])

    async def _write_inbound(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """One edge per stored link that names this document, written as its target appears.

        The search is lexical and bounded, and both properties are deliberate. Lexical, because
        the text being looked for is the slug itself and BM25 over stored chunk text is the one
        index manicule already keeps that can find it without reading the corpus. Bounded,
        because a slug made of common words would otherwise match a large fraction of a corpus —
        and the bound costs nothing that is not recovered: a linking document outside it acquires
        its edge the next time it is scanned.

        Every candidate is **re-scanned with the real rules** before an edge is written. BM25
        matches a chunk that merely contains the words, so treating a hit as a link would invent
        edges from prose that happened to mention a document's name.
        """
        entry = _entry_of(chunks)
        slug = slug_of(document)
        words = slug.split("-")
        if (
            not self._resolvable(document)
            or not self._config.inbound_limit
            or len(words) < MIN_SEARCHABLE_SLUG_TOKENS
        ):
            return
        # The slug's **words**, not the slug. A link is written `[[project_backoff]]`, and the
        # lexical index tokenizes that to `project` and `backoff` — so searching for the slug as
        # one string is a phrase with a separator in it, which matches nothing. Searching the
        # words is also what makes the hyphen and underscore spellings one query rather than
        # two: both tokenize identically, which is the same normalization `normalize` performs,
        # arrived at by the index rather than by this module.
        for candidate in await self._store.search_lexical(
            " ".join(words), self._config.inbound_limit, self._scope()
        ):
            chunk = candidate.chunk
            if chunk.document_id == document.id:
                continue
            for link in links_in(chunk.text):
                if link.target == slug:
                    await self._relate(chunk.id, entry, _RELATION_OF[link.shape])

    # --- internals ---------------------------------------------------------------------------

    async def _claim(self, slugs: dict[str, str], slug: str, document_id: str) -> None:
        """Record ``document_id`` as what ``slug`` resolves to, unless something live holds it.

        **Not a plain ``setdefault``**, and the difference is a rename. The index is built once
        per pipeline and kept current incrementally, so an entry made at startup outlives the
        document it names: delete ``a.md``, write the same fact as ``b.md``, and a
        ``setdefault`` keeps pointing ``b`` at the deleted document — every link to it silently
        resolves to nothing until the process restarts. Not a plain assignment either, because
        two *live* documents can share a stem and last-writer-wins would make the graph depend
        on the order a sync happened to reach them.

        So the incumbent is displaced only when it is gone. The liveness question costs one small
        query, and only when a slug is already taken by a different document.
        """
        incumbent = slugs.get(slug)
        if incumbent == document_id:
            return
        if incumbent is not None and await self._entry_chunk(incumbent) is not None:
            return
        slugs[slug] = document_id

    def _resolvable(self, document: Document) -> bool:
        """Whether a link may name ``document`` at all.

        Asked in both directions and from one place, because the two would otherwise be able to
        disagree: a document registered in the index as a target but skipped by the inbound search
        would be reachable by links written after it and not by links written before it, which is
        a graph whose shape depends on ingest order.
        """
        return not self._config.sources or document.source in self._config.sources

    def _scope(self) -> Filter | None:
        """The sources a link may resolve into, or ``None`` for every source.

        ``None`` rather than an empty :class:`~manicule.core.retrieval.Filter`, because an empty
        filter restricts nothing and constructing one to say so would be a second spelling of the
        same answer.
        """
        from manicule.core.retrieval import Filter  # noqa: PLC0415 - one narrow import, used twice

        if not self._config.sources:
            return None
        return Filter(
            workspace_ids=frozenset({self._store.workspace_id}),
            sources=frozenset(self._config.sources),
        )

    async def _index(self) -> dict[str, str]:
        """Normalized slug to document id, built once and then kept current incrementally.

        Paged rather than read whole, because ``list_documents`` takes a limit and a corpus is
        not bounded. Built lazily rather than at construction: a pipeline that ingests nothing
        should pay nothing, and construction happens on every start.

        **Two documents can share a slug**, because a slug is a filename stem and stems are not
        unique across sources or across the directories inside one. The first in the store's own
        order wins, which makes the choice deterministic rather than right — ``sources`` is how an
        installation narrows the space so the question does not arise.
        """
        if self._slugs is not None:
            return self._slugs
        built: dict[str, str] = {}
        offset = 0
        scope = self._scope()
        while True:
            page = await self._store.list_documents(scope, limit=_INDEX_PAGE, offset=offset)
            if not page:
                break
            for document in page:
                built.setdefault(slug_of(document), document.id)
            offset += len(page)
        self._slugs = built
        return built

    async def _entry_chunk(self, document_id: str) -> str | None:
        """The chunk an edge into ``document_id`` points at: its first, or ``None`` if it has none.

        **A link addresses a document and an edge addresses a chunk**, so one of them has to be
        chosen. The first chunk is the document's opening, which for a corpus of one
        self-contained fact per file is the fact — and it is the only choice that does not depend
        on the link's wording, so two links to one document from two places agree about where
        they point.

        Looked up rather than cached, because a chunk id is derived from its text and a re-ingest
        that changed a document's opening would leave a cached id naming a chunk that no longer
        exists. The cost is one small query per distinct target per document.
        """
        chunks = await self._store.document_chunks(document_id)
        return _entry_of(chunks) if chunks else None

    async def _relate(self, source: str, target: str, relation: ChunkRelationType) -> None:
        """Write one edge. A race at either end **fails the hook** rather than being swallowed.

        ``relate`` refuses a chunk that is not live in this workspace, and between reading a
        target's chunks and writing the edge a concurrent sync can have replaced them. Catching
        that and carrying on reads as tolerance and is a way of losing an edge for good: the hook
        would return normally, the pipeline would stamp this document's relation lineage as
        current, and the repair selects on that lineage — so the document would never be looked
        at again. The **inbound** direction makes it worse, because there the edge belongs to
        some *other* document that already carries a current fingerprint of its own and would not
        be selected either.

        So the exception propagates. The pipeline treats a raising ``after_store`` as this hook's
        problem — the document stays published, the failure is recorded in its metadata — and,
        crucially, leaves ``relation_fp`` untouched, so the next repair picks the document up and
        writes the edges against chunks that are current by then. Losing a scan is recoverable;
        losing an edge while claiming the scan happened is not.
        """
        if source == target:
            return
        await self._store.relate(source, target, relation)


class WikilinkPlugin:
    """The plugin object the entry point resolves to."""

    manifest = PluginManifest(
        name="wikilinks",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="Turns [[wikilinks]] in markdown into typed chunk relations.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.MIDDLEWARE.named("wikilinks"),
            build_middleware,
            config_model=WikilinkConfig,
            summary="Writes chunk relations for [[wikilinks]], in both directions.",
        )


def build_middleware(context: BuildContext) -> WikilinkMiddleware:
    """Factory. Resolves the one dependency, and refuses a store that cannot carry edges.

    Public, unlike the example plugin's, because the refusal below is behavior with a test of its
    own: a store that cannot carry edges has to be told so at construction, naming what is
    missing, rather than producing an attribute error somewhere unrelated much later.

    Refused rather than duck-typed: this writes rows keyed on chunks in a table with foreign
    keys, and guessing that an unknown store means the same thing by ``relate`` is how edges end
    up somewhere nobody named. The message says which methods are missing, because "a store was
    the wrong type" is not something an operator can act on.
    """
    config = context.config
    if not isinstance(config, WikilinkConfig):  # pragma: no cover - the container guarantees it
        config = WikilinkConfig()
    store = context.components.get(keys.DOC_STORE)
    if not isinstance(store, RelationCorpus):
        missing = sorted(
            name
            for name in (
                "workspace_id",
                "document_chunks",
                "list_documents",
                "search_lexical",
                "relate",
            )
            if getattr(store, name, None) is None
        )
        msg = (
            f"the configured document store is a {type(store).__name__}, which does not offer "
            f"{missing}. The wikilinks middleware writes chunk_relations rows and reads chunk "
            f"text back, so a store that cannot do both has nowhere to put a graph."
        )
        raise ConfigError(msg)
    return WikilinkMiddleware(store, config)


PLUGIN = WikilinkPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with the
# protocols it implements — including the extractor one, which is what the pipeline looks for
# when it decides whose rules to record against every document. Callables rather than instances,
# because this middleware takes a store and there is none at import time: what has to be checked
# is that the constructor produces something of each shape.
_plugin: Plugin = PLUGIN
_hook: Callable[[RelationCorpus, WikilinkConfig], Middleware] = WikilinkMiddleware
_extractor: Callable[[RelationCorpus, WikilinkConfig], ChunkRelationExtractor] = WikilinkMiddleware


__all__ = [
    "EXTRACTOR",
    "PLUGIN",
    "SOURCES",
    "RelationCorpus",
    "WikilinkConfig",
    "WikilinkMiddleware",
    "WikilinkPlugin",
    "build_middleware",
    "rules_digest",
    "slug_of",
]
