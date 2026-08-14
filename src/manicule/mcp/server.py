"""The MCP server: twenty-eight tools, each a few lines over the application service.

FastMCP derives every tool's schema from the function's type hints and its description from
the docstring, so what an assistant sees is what the signature says. There is no protocol
plumbing here to keep in step with the specification, and no second description of an
operation to drift from the first.

**Every tool returns the same envelope the command line prints under ``--json``.** Same
builder, same payload models, same error mapping — see :mod:`manicule.app.dispatch`. A
consumer that can read one surface can read the other, and
``tests/app/test_surface_parity.py`` fails if that stops being true.

**Failures are results, not exceptions.** A tool that raised would give an assistant a
transport-level error with no shape to it. ``ok: false`` with a type, a message and a hint is
something it can act on — retry, ask the operator, or say plainly what went wrong. Defects
still propagate: a bug is not a result.

**The default transport is stdio, which binds nothing.** There is no address to get wrong on
the path everybody uses. The HTTP transport exists, goes through
:func:`~manicule.app.bind.resolve_bind` like every other server, and refuses a non-loopback
bind that was not asked for three separate times.

**Every tool says what it does to the installation**, in the four hints ``tools/list``
carries — see :func:`hints`. They are a description, never a permission: a client decides what
it will call, and nothing here consults them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from manicule.app.dispatch import run_op
from manicule.core.version import CORE_VERSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from manicule.app.results import Payload
    from manicule.app.service import ApplicationService

SERVER_NAME = "manicule"

INSTRUCTIONS = """\
Search and question-answering over a self-hosted document index.

Every tool returns the same envelope: `ok`, `op`, `workspace`, and then `data` or `error`.
Read `ok` first. Everything is scoped to one workspace and nothing crosses between them.

`search` ranks passages and asks no model anything; `ask` answers in prose with citations.
Every citation resolves to a real location in a real document, and one that could not be
verified is deleted rather than shown — so an answer with no citations means nothing could be
verified, not that nothing was found.

## Scope every question to a collection, and resolve the collection first

1. `collection_list` — find the collection by name. Keep both fields it reports: `id` is
   stable and is what `collection_counts` takes; `name` is what `search` and `ask` take as a
   scope.
2. `collection_counts(collection_id)` — the current document and chunk totals. They are
   counted from membership when you ask, which is why they are a separate call and why no row
   in `collection_list` carries a total: a remembered total reports the day it was written.
3. `search(query, collections=[name], limit=5)` — name the scope on **every** call. Nothing is
   remembered between calls. `data.collections` repeats the scope the search actually ran
   under; read it rather than assuming the argument arrived.

**A scope that fails is a refusal, not permission to widen.** A name this workspace does not
have comes back `ok: false` with `UnknownEntityError`, and no search runs — deliberately, since
a restriction that silently vanished would return the whole workspace, ranked and plausible.
Correct the name or say you cannot answer. Do not retry the same query without `collections`.

**Retrieving nothing is not proof that there is nothing.** `search` returns the top `limit`
passages of one ranking, so an empty or weak result means the top of that ranking held nothing,
not that the corpus does not. Say "nothing in <collection> supports this", never "there is no
such thing"; `collection_counts` is what tells you how much you did not look at.

**Keep retrieval small.** Three searches at `limit=4` or `5` answers most questions. Add a
fourth only to close a gap you can name. Paraphrase what you cite and keep its title, URI and
heading path — copying long passages forward spends context on text you have already read.
"""


def hints(*, reads: bool, removes: bool, repeatable: bool, reaches_out: bool) -> ToolAnnotations:
    """The four hints ``tools/list`` carries for one tool, from four questions about it.

    The questions are asked in English so that they can be answered from the tool's behavior
    rather than from its name, and translated here — once — into the specification's own
    vocabulary. Every argument is required and none has a default: a tool added tomorrow has to
    answer all four, and a default is how the wrong answer gets given by nobody.

    **These are hints. They are not authorization.** The specification says so and this server
    behaves accordingly: nothing reads them back, no tool is gated on them, and a client that
    ignores them reaches exactly what it reached before. What they buy is an operator being
    able to approve `search` without also approving `document_delete` — see
    ``docs/surfaces.md`` §4.1.

    Args:
        reads: The call leaves the index, the configuration and the installation as it found
            them. Retrieval's one append to ``query_logs`` is the exception the whole project
            already makes for it (:data:`~manicule.app.dispatch.READ_ONLY_OPS`): it records
            that a read happened rather than changing what any read reports.
        removes: The call can remove or overwrite something it does not carry enough
            information to put back. ``collection_remove`` is *not* one — it names the
            documents, so ``collection_add`` with the same ones restores it — while
            ``collection_update`` is, because the description it overwrites is not in the call.
        repeatable: Calling it again with the same arguments changes nothing further.
        reaches_out: It can reach something outside this installation — a remote system, or a
            part of this machine manicule does not own.

    Returns:
        The annotations, ready to hand to ``@mcp.tool``.
    """
    return ToolAnnotations(
        readOnlyHint=reads,
        destructiveHint=removes,
        idempotentHint=repeatable,
        openWorldHint=reaches_out,
    )


READS = hints(reads=True, removes=False, repeatable=True, reaches_out=False)
"""A tool that reads this installation and reaches nothing beyond it.

Named because twelve tools are exactly this and repeating the four arguments twelve times
would invite one of them to be edited alone. It is a *value*, not a list of tool names: which
tools carry it is written at the registrations and nowhere else, so there is no second table to
fall out of step with them.
"""

TOOL_NAMES: tuple[str, ...] = (
    "ask",
    "search",
    "index_path",
    "document_list",
    "document_get",
    "document_delete",
    "document_reindex",
    "index_status",
    "stats",
    "doctor",
    "connector_list",
    "connector_sync",
    "config_get",
    "config_set",
    "workspace_list",
    "workspace_switch",
    "plugin_list",
    "plugin_add",
    "plugin_remove",
    "collection_create",
    "collection_list",
    "collection_rename",
    "collection_update",
    "collection_delete",
    "collection_add",
    "collection_remove",
    "collection_documents",
    "collection_counts",
)
"""The tool surface, named once.

Here as data as well as decorators so that "the server offers exactly these" is a test rather
than a count somebody keeps in their head.
"""


def _register_collections(
    mcp: FastMCP,
    service: ApplicationService,
    dispatch: Callable[[str, Callable[[], Awaitable[Payload]]], Awaitable[dict[str, Any]]],
) -> tuple[Any, ...]:
    """Register the collection tools and hand back what was registered.

    Split out of :func:`build_server` because nine more tools pushed one function past the
    point where a reader can hold it, not because collections are a different kind of thing.
    The registered functions are *returned* rather than dropped, so they reach the same
    surface-versus-``TOOL_NAMES`` comparison every other tool is held to — a group registered
    here and forgotten there would be exactly the drift that check exists to catch.

    Note the one operation deliberately absent: there is no tool that deletes documents left
    in no collection. It destroys data, so it stays on the command line with the rest of that
    class -- ``reset-index``, ``backup``, ``import`` -- where a person is present.
    """

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=False, reaches_out=False))
    async def collection_create(name: str, description: str | None = None) -> dict[str, Any]:
        """Create a named set of documents.

        A collection groups documents that are already indexed. It never copies them: a
        document in two collections is still one document with one set of embeddings.

        Args:
            name: What to call it. A name already in use is refused rather than merged.
            description: What the collection is for.
        """
        return await dispatch(
            "collection_create", lambda: service.collection_create(name, description=description)
        )

    @mcp.tool(annotations=READS)
    async def collection_list() -> dict[str, Any]:
        """List every collection in this workspace, with the rule each one carries.

        **Start here whenever a question names a collection.** Each row carries both identities
        and they are not interchangeable: `id` is stable and is what `collection_counts` takes,
        `name` is what `search` and `ask` take as a scope.

        No row carries a document or chunk total. That is `collection_counts`, computed when
        you ask it, because a total remembered here would report the day it was written.
        """
        return await dispatch("collection_list", service.collection_list)

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=True, reaches_out=False))
    async def collection_rename(collection_id: str, name: str) -> dict[str, Any]:
        """Rename a collection. Nothing is re-indexed and no membership moves.

        Args:
            collection_id: The collection to rename.
            name: The new name. Another collection already using it is refused.
        """
        return await dispatch(
            "collection_rename", lambda: service.collection_rename(collection_id, name)
        )

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=True, reaches_out=False))
    async def collection_update(collection_id: str, description: str) -> dict[str, Any]:
        """Set a collection's description, leaving its membership alone.

        The write is a set, not a merge. ``description`` is required for that reason: as an
        optional argument, calling this tool without one erased the description rather than
        leaving it alone, which is a destructive default nobody would ask for.

        Args:
            collection_id: The collection to describe.
            description: What it is for. An empty string clears it.
        """
        return await dispatch(
            "collection_update",
            lambda: service.collection_update(collection_id, description=description),
        )

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=False, reaches_out=False))
    async def collection_delete(collection_id: str) -> dict[str, Any]:
        """Delete a collection. **The documents in it are untouched.**

        A collection is a grouping, so deleting one removes the grouping and stops there.
        Documents left in no collection can be listed, and removed, from the command line.

        Args:
            collection_id: The collection to delete.
        """
        return await dispatch("collection_delete", lambda: service.collection_delete(collection_id))

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=True, reaches_out=False))
    async def collection_add(collection_id: str, document_ids: list[str]) -> dict[str, Any]:
        """Add documents to a collection.

        Membership is metadata. Nothing is re-parsed, re-chunked or re-embedded, and a
        document already in the collection is not added twice.

        Args:
            collection_id: The collection to add to.
            document_ids: Documents to add, as ``document_list`` or a citation reported them.
        """
        return await dispatch(
            "collection_add", lambda: service.collection_add(collection_id, tuple(document_ids))
        )

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=True, reaches_out=False))
    async def collection_remove(collection_id: str, document_ids: list[str]) -> dict[str, Any]:
        """Remove documents from a collection. The documents themselves survive.

        Not annotated as removing anything, and that is the annotation being accurate rather
        than generous: the call names the documents, so `collection_add` with the same ones
        puts the membership back. `collection_delete` is the one that cannot be undone from
        its own arguments.

        Args:
            collection_id: The collection to remove from.
            document_ids: Documents to drop from it.
        """
        return await dispatch(
            "collection_remove",
            lambda: service.collection_remove(collection_id, tuple(document_ids)),
        )

    @mcp.tool(annotations=READS)
    async def collection_documents(
        collection_id: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List a collection's documents, those added by hand and those a rule selects alike.

        **A page, not a census.** `count` is how many this page holds, so paging until a short
        page comes back is the only way to see everything here. For "how many are there", ask
        `collection_counts`, which counts the whole membership in one call.

        Args:
            collection_id: The collection to read.
            limit: Page size.
            offset: How many to skip.
        """
        return await dispatch(
            "collection_documents",
            lambda: service.collection_documents(collection_id, limit=limit, offset=offset),
        )

    @mcp.tool(annotations=READS)
    async def collection_counts(collection_id: str) -> dict[str, Any]:
        """Count a collection's documents and chunks, as they are now.

        **The authoritative answer to how much a collection holds**, and the only one. Both
        numbers are computed from membership on the call, which is why `collection_list`
        reports no total: one remembered there would report the day it was written. A
        rule-driven collection has no stored membership at all, so there is nothing to
        remember even in principle.

        Ask this rather than counting a `collection_documents` page or the hits a `search`
        returned. A page is a page, and a search returns its top `limit` — neither is a total,
        and treating one as a total understates a corpus silently.

        Args:
            collection_id: The collection to count, as `collection_list` reported its `id`.
        """
        return await dispatch("collection_counts", lambda: service.collection_counts(collection_id))

    return (
        collection_create,
        collection_list,
        collection_rename,
        collection_update,
        collection_delete,
        collection_add,
        collection_remove,
        collection_documents,
        collection_counts,
    )


def build_server(service: ApplicationService) -> FastMCP:
    """Register every tool against ``service`` and return the server.

    Takes the service rather than building one, so the suites drive the real FastMCP tool
    manager against a fake backend — which is what makes the parity test meaningful. A server
    that constructed its own runtime could only be tested by starting a whole manicule.
    """
    mcp: FastMCP = FastMCP(
        name=SERVER_NAME,
        version=CORE_VERSION,
        instructions=INSTRUCTIONS,
    )

    async def dispatch(op: str, call: Callable[[], Awaitable[Payload]]) -> dict[str, Any]:
        return (await run_op(op, service.workspace, call)).as_json()

    # --- questions ------------------------------------------------------------------------

    # Not read-only, for two reasons that are each sufficient: with a `conversation_id` it
    # persists this turn, and the model it calls may be a provider on somebody else's machine.
    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=False, reaches_out=True))
    async def ask(
        question: str,
        *,
        profile: str | None = None,
        limit: int | None = None,
        sources: list[str] | None = None,
        collections: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question from the indexed corpus, with citations that resolve.

        Args:
            question: What to answer. A natural-language question, not a keyword query.
            profile: ``fast``, ``balanced`` or ``precise``. Omit to use the configured one.
            limit: How many passages to retrieve before answering.
            sources: Restrict to these source names. Omit to search everything.
            collections: Restrict to these collections, named as ``collection_list`` reports
                them. Several union. A name that is not a collection here refuses the call
                rather than answering over the whole workspace.
            conversation_id: Continue an existing conversation, and persist this turn to it.

        Returns:
            An envelope whose ``data`` carries the answer text, its citations, the retrieval
            confidence and whether the corpus was consulted at all. ``ungrounded`` means
            passages were found and none survived verification — a materially weaker answer.
            ``explicit_definition`` is ``true`` when the question asked what a term means and
            the corpus's own definition of it is in ``expansions`` and was in the context; read
            that boolean rather than parsing ``confidence_reason``, and read it *beside*
            ``confidence`` rather than as a larger one — it moves no number.
        """
        return await dispatch(
            "ask",
            lambda: service.ask(
                question,
                profile=profile,
                limit=limit,
                sources=tuple(sources or ()),
                collections=tuple(collections or ()),
                conversation_id=conversation_id,
            ),
        )

    @mcp.tool(annotations=READS)
    async def search(
        query: str,
        *,
        limit: int = 10,
        profile: str | None = None,
        sources: list[str] | None = None,
        media_types: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Rank passages for a query without asking a model anything.

        Cheaper and more predictable than ``ask`` when you want the source material rather
        than prose about it, and the right tool when you intend to read the passages yourself.

        **Name the scope on every call.** Nothing is remembered between calls, so a `search`
        with no `collections` searches the whole workspace however the last one was scoped.
        `data.collections` repeats the scope this search ran under — read it rather than
        assuming the argument arrived.

        **This returns the top `limit` passages of one ranking, which is not a census of the
        corpus.** An empty or weak result means the top of that ranking held nothing, not that
        the corpus does not hold it. Say so that way, and use `collection_counts` for how much
        was not looked at. Small limits — 4 or 5 — answer most questions and leave room to ask
        another.

        Reading is all this does, with one exception stated rather than hidden: the retrieval
        is recorded in the local query log, which notes that a search happened and changes
        nothing any search, answer or listing reports.

        Args:
            query: What to search for.
            limit: How many passages to return.
            profile: ``fast``, ``balanced`` or ``precise``.
            sources: Restrict to these source names.
            media_types: Restrict to these IANA media types, e.g. ``application/pdf``.
            collections: Restrict to these collections, named as ``collection_list`` reports
                them. Several union; a collection combined with a source keeps only what is in
                both. A name that is not a collection here refuses the call rather than
                searching the whole workspace.

        Returns:
            An envelope whose ``data.hits`` are ranked passages, each with its document, its
            anchor and the score every pipeline stage gave it. ``data.collections`` repeats
            the scope the search ran under. ``data.explicit_definition`` is ``true`` when the
            query asked what a term means and the corpus's own definition of it is in
            ``data.expansions`` and reached the results; read that boolean rather than parsing
            ``confidence_reason``, and read it *beside* ``confidence`` rather than as a larger
            one — it moves no number.
        """
        return await dispatch(
            "search",
            lambda: service.search(
                query,
                limit=limit,
                profile=profile,
                sources=tuple(sources or ()),
                media_types=tuple(media_types or ()),
                collections=tuple(collections or ()),
            ),
        )

    # --- ingest ---------------------------------------------------------------------------

    # `reaches_out`, because the tree it walks is a part of this machine manicule does not own
    # and cannot enumerate in advance. No network is involved; the hint is not about networks.
    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=False, reaches_out=True))
    async def index_path(
        path: str,
        source: str = "local",
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Index a file or a directory tree from this machine's filesystem.

        Args:
            path: What to index. A directory is walked recursively; version-control
                directories and tool caches are skipped.
            source: The source name documents are recorded under. It is part of their
                identity, so changing it re-indexes rather than updates.
            limit: Stop after this many discovered documents.
            force: Re-parse documents whose bytes have not changed. Use after a parser change.

        Returns:
            An envelope whose ``data`` carries the run's counters, including ``by_status`` —
            where an outcome that is neither an ingest nor a failure, such as a PDF with no
            extractable text, is visible rather than folded into one of the two.
        """
        return await dispatch(
            "index_path",
            lambda: service.index_path(path, source=source, limit=limit, force=force),
        )

    # --- documents ------------------------------------------------------------------------

    @mcp.tool(annotations=READS)
    async def document_list(
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        """List indexed documents in this workspace, newest first.

        Args:
            limit: Page size.
            offset: How many to skip.
            source: Restrict to one source name.
            media_type: Restrict to one IANA media type.
        """
        return await dispatch(
            "document_list",
            lambda: service.document_list(
                limit=limit, offset=offset, source=source, media_type=media_type
            ),
        )

    @mcp.tool(annotations=READS)
    async def document_get(document_id: str, chunks: bool = False) -> dict[str, Any]:
        """Read one document's metadata, and optionally every chunk it was split into.

        Args:
            document_id: The id ``document_list`` or a citation reported.
            chunks: Include the stored chunks. They carry the text a citation quotes.
        """
        return await dispatch(
            "document_get", lambda: service.document_get(document_id, chunks=chunks)
        )

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=False, reaches_out=False))
    async def document_delete(document_id: str, hard: bool = False) -> dict[str, Any]:
        """Remove a document from the index.

        Args:
            document_id: What to remove.
            hard: Delete outright instead of moving it to the trash. A soft delete can be
                restored; a hard one cannot.
        """
        return await dispatch(
            "document_delete", lambda: service.document_delete(document_id, hard=hard)
        )

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=True, reaches_out=False))
    async def document_reindex(document_id: str) -> dict[str, Any]:
        """Re-parse one document from the bytes ingest retained. Touches no network.

        The repair for a parser fix or a chunker change. Chunk ids are derived from content,
        so a chunk that survives unchanged keeps its vector.

        Args:
            document_id: The document to re-parse. It must be live: a document in the trash
                has to be restored first.
        """
        return await dispatch("document_reindex", lambda: service.document_reindex(document_id))

    # --- state ----------------------------------------------------------------------------

    @mcp.tool(annotations=READS)
    async def index_status() -> dict[str, Any]:
        """Report what is in the index and what it was built with.

        Counts by document status, the embedding and chunking fingerprints the index is
        committed to, and the database's schema revision.
        """
        return await dispatch("index_status", service.index_status)

    @mcp.tool(annotations=READS)
    async def stats() -> dict[str, Any]:
        """Count documents and chunks, grouped by source, media type and status."""
        return await dispatch("stats", service.stats)

    # Read-only because `fix` is not on this surface: the repairs that write to the machine and
    # may fetch from the network are passed by the command line alone, and
    # `tests/app/test_surface_parity.py` holds that line.
    @mcp.tool(annotations=READS)
    async def doctor() -> dict[str, Any]:
        """Check configuration, plugins, storage, the index and the network bind.

        Builds nothing expensive: no model runtime is loaded and no document is read, so this
        is safe to run on an installation that is not working.
        """
        return await dispatch("doctor", service.doctor)

    # --- connectors -----------------------------------------------------------------------

    @mcp.tool(annotations=READS)
    async def connector_list() -> dict[str, Any]:
        """List configured sources, with what each one's last sync recorded."""
        return await dispatch("connector_list", service.connector_list)

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=False, reaches_out=True))
    async def connector_sync(name: str, limit: int | None = None) -> dict[str, Any]:
        """Run one configured connector, ingesting what changed since its watermark.

        Args:
            name: The instance name from the ``connectors`` section of configuration.
            limit: Stop after this many discovered documents.
        """
        return await dispatch("connector_sync", lambda: service.connector_sync(name, limit=limit))

    # --- configuration --------------------------------------------------------------------

    @mcp.tool(annotations=READS)
    async def config_get(key: str = "") -> dict[str, Any]:
        """Read configuration, with every credential masked.

        Args:
            key: A dotted key such as ``rag.profile``. Omit for the whole tree.
        """
        return await dispatch("config_get", lambda: service.config_get(key))

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=True, reaches_out=False))
    async def config_set(key: str, value: str) -> dict[str, Any]:
        """Write one setting to the config file, validating the whole tree first.

        Args:
            key: A dotted key such as ``rag.profile``.
            value: Parsed as JSON when it parses — ``false``, ``12``, ``["a"]`` — and kept as
                a string when it does not, so ``qwen2.5:14b`` needs no quoting.

        A credential is refused rather than written: those belong in the environment, where
        they are not copied into backups and exports.
        """
        return await dispatch("config_set", lambda: service.config_set(key, value))

    # --- workspaces -----------------------------------------------------------------------

    @mcp.tool(annotations=READS)
    async def workspace_list() -> dict[str, Any]:
        """List the workspaces this installation knows about, and say which is active.

        Document counts are reported for the active workspace only. This process is scoped to
        one tenant and cannot read another's rows, including to count them.
        """
        return await dispatch("workspace_list", service.workspace_list)

    @mcp.tool(annotations=hints(reads=False, removes=True, repeatable=True, reaches_out=False))
    async def workspace_switch(name: str, create: bool = False) -> dict[str, Any]:
        """Record a different active workspace. It takes effect at the next start.

        Deliberately not immediate: a process holding some handles scoped to the old workspace
        and some to the new one is the state in which a cross-tenant read stops being
        impossible.

        Args:
            name: The workspace to make active.
            create: Accept a name that does not exist yet.
        """
        return await dispatch(
            "workspace_switch", lambda: service.workspace_switch(name, create=create)
        )

    # --- plugins --------------------------------------------------------------------------

    # Read-only and `reaches_out` together, which is the pair a name would have got wrong:
    # `registry=True` fetches the community listing over the network. It writes nothing either
    # way, so the first hint is true and the fourth is what says a call may leave this machine.
    @mcp.tool(annotations=hints(reads=True, removes=False, repeatable=True, reaches_out=True))
    async def plugin_list(registry: bool = False) -> dict[str, Any]:
        """List installed plugins and the components each one registers.

        Args:
            registry: Also fetch the community listing. Only consulted when
                ``plugins.allow_install`` is on, because a plugin runs with this process's
                full authority and browsing a catalog of them is opt-in.
        """
        return await dispatch("plugin_list", lambda: service.plugin_list(registry=registry))

    # `reaches_out`, which the name argues against and the code settles: a name that is not
    # installed sends `plugin_add` to the community registry to find out whether it exists, so
    # that the refusal can name the command that would install it.
    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=True, reaches_out=True))
    async def plugin_add(name: str) -> dict[str, Any]:
        """Enable an installed plugin.

        This never installs anything. Installing a plugin fetches and runs code with this
        process's full authority, and a tool an assistant can call unattended must not be able
        to do that. A plugin that is not installed is reported with the command that would
        install it.

        Args:
            name: The plugin's name, as ``plugin_list`` reports it.
        """
        return await dispatch("plugin_add", lambda: service.plugin_add(name))

    @mcp.tool(annotations=hints(reads=False, removes=False, repeatable=True, reaches_out=False))
    async def plugin_remove(name: str) -> dict[str, Any]:
        """Disable a plugin. The distribution stays installed and is not touched.

        Args:
            name: The plugin's name, as ``plugin_list`` reports it.
        """
        return await dispatch("plugin_remove", lambda: service.plugin_remove(name))

    collections = _register_collections(mcp, service, dispatch)

    # Every tool, named twice — once by its decorator and once in `TOOL_NAMES` — and the two
    # are compared here rather than in a test. A tool added without being listed would
    # otherwise be a surface nothing describes, and a name listed without a tool would be a
    # tool a client is told about and cannot call. Both are startup failures.
    registered = {
        # `__name__`, because FastMCP derives a tool's name from its function's and the
        # decorator hands the function straight back. Reading the name off the registry
        # instead would need an await, and this is a wiring check rather than a query.
        tool.__name__
        for tool in (
            ask,
            search,
            index_path,
            document_list,
            document_get,
            document_delete,
            document_reindex,
            index_status,
            stats,
            doctor,
            connector_list,
            connector_sync,
            config_get,
            config_set,
            workspace_list,
            workspace_switch,
            plugin_list,
            plugin_add,
            plugin_remove,
            *collections,
        )
    }
    if registered != set(TOOL_NAMES):
        missing = sorted(set(TOOL_NAMES) - registered)
        extra = sorted(registered - set(TOOL_NAMES))
        msg = (
            f"the MCP tool surface and TOOL_NAMES disagree. Registered but unlisted: "
            f"{extra or 'none'}. Listed but not registered: {missing or 'none'}."
        )
        raise AssertionError(msg)
    return mcp


__all__ = ["INSTRUCTIONS", "READS", "SERVER_NAME", "TOOL_NAMES", "build_server", "hints"]
