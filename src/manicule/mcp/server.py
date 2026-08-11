"""The MCP server: nineteen tools, each a few lines over the application service.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from manicule.app.dispatch import run_op
from manicule.core.version import CORE_VERSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from manicule.app.results import Payload
    from manicule.app.service import ApplicationService

SERVER_NAME = "manicule"

INSTRUCTIONS = """\
Search and question-answering over a self-hosted document index.

Start with `search` when you want passages and `ask` when you want an answer with citations.
Every citation resolves to a real location in a real document, and one that could not be
verified is deleted rather than shown — so an answer with no citations means nothing could be
verified, not that nothing was found.

Every tool returns the same envelope: `ok`, `op`, `workspace`, and then `data` or `error`.
Read `ok` first. Everything is scoped to one workspace and nothing crosses between them.
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
)
"""The tool surface, named once.

Here as data as well as decorators so that "the server offers exactly these" is a test rather
than a count somebody keeps in their head.
"""


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

    @mcp.tool
    async def ask(
        question: str,
        profile: str | None = None,
        limit: int | None = None,
        sources: list[str] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question from the indexed corpus, with citations that resolve.

        Args:
            question: What to answer. A natural-language question, not a keyword query.
            profile: ``fast``, ``balanced`` or ``precise``. Omit to use the configured one.
            limit: How many passages to retrieve before answering.
            sources: Restrict to these source names. Omit to search everything.
            conversation_id: Continue an existing conversation, and persist this turn to it.

        Returns:
            An envelope whose ``data`` carries the answer text, its citations, the retrieval
            confidence and whether the corpus was consulted at all. ``ungrounded`` means
            passages were found and none survived verification — a materially weaker answer.
        """
        return await dispatch(
            "ask",
            lambda: service.ask(
                question,
                profile=profile,
                limit=limit,
                sources=tuple(sources or ()),
                conversation_id=conversation_id,
            ),
        )

    @mcp.tool
    async def search(
        query: str,
        limit: int = 10,
        profile: str | None = None,
        sources: list[str] | None = None,
        media_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Rank passages for a query without asking a model anything.

        Cheaper and more predictable than ``ask`` when you want the source material rather
        than prose about it, and the right tool when you intend to read the passages yourself.

        Args:
            query: What to search for.
            limit: How many passages to return.
            profile: ``fast``, ``balanced`` or ``precise``.
            sources: Restrict to these source names.
            media_types: Restrict to these IANA media types, e.g. ``application/pdf``.

        Returns:
            An envelope whose ``data.hits`` are ranked passages, each with its document, its
            anchor and the score every pipeline stage gave it.
        """
        return await dispatch(
            "search",
            lambda: service.search(
                query,
                limit=limit,
                profile=profile,
                sources=tuple(sources or ()),
                media_types=tuple(media_types or ()),
            ),
        )

    # --- ingest ---------------------------------------------------------------------------

    @mcp.tool
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

    @mcp.tool
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

    @mcp.tool
    async def document_get(document_id: str, chunks: bool = False) -> dict[str, Any]:
        """Read one document's metadata, and optionally every chunk it was split into.

        Args:
            document_id: The id ``document_list`` or a citation reported.
            chunks: Include the stored chunks. They carry the text a citation quotes.
        """
        return await dispatch(
            "document_get", lambda: service.document_get(document_id, chunks=chunks)
        )

    @mcp.tool
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

    @mcp.tool
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

    @mcp.tool
    async def index_status() -> dict[str, Any]:
        """Report what is in the index and what it was built with.

        Counts by document status, the embedding and chunking fingerprints the index is
        committed to, and the database's schema revision.
        """
        return await dispatch("index_status", service.index_status)

    @mcp.tool
    async def stats() -> dict[str, Any]:
        """Count documents and chunks, grouped by source, media type and status."""
        return await dispatch("stats", service.stats)

    @mcp.tool
    async def doctor() -> dict[str, Any]:
        """Check configuration, plugins, storage, the index and the network bind.

        Builds nothing expensive: no model runtime is loaded and no document is read, so this
        is safe to run on an installation that is not working.
        """
        return await dispatch("doctor", service.doctor)

    # --- connectors -----------------------------------------------------------------------

    @mcp.tool
    async def connector_list() -> dict[str, Any]:
        """List configured sources, with what each one's last sync recorded."""
        return await dispatch("connector_list", service.connector_list)

    @mcp.tool
    async def connector_sync(name: str, limit: int | None = None) -> dict[str, Any]:
        """Run one configured connector, ingesting what changed since its watermark.

        Args:
            name: The instance name from the ``connectors`` section of configuration.
            limit: Stop after this many discovered documents.
        """
        return await dispatch("connector_sync", lambda: service.connector_sync(name, limit=limit))

    # --- configuration --------------------------------------------------------------------

    @mcp.tool
    async def config_get(key: str = "") -> dict[str, Any]:
        """Read configuration, with every credential masked.

        Args:
            key: A dotted key such as ``rag.profile``. Omit for the whole tree.
        """
        return await dispatch("config_get", lambda: service.config_get(key))

    @mcp.tool
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

    @mcp.tool
    async def workspace_list() -> dict[str, Any]:
        """List the workspaces this installation knows about, and say which is active.

        Document counts are reported for the active workspace only. This process is scoped to
        one tenant and cannot read another's rows, including to count them.
        """
        return await dispatch("workspace_list", service.workspace_list)

    @mcp.tool
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

    @mcp.tool
    async def plugin_list(registry: bool = False) -> dict[str, Any]:
        """List installed plugins and the components each one registers.

        Args:
            registry: Also fetch the community listing. Only consulted when
                ``plugins.allow_install`` is on, because a plugin runs with this process's
                full authority and browsing a catalogue of them is opt-in.
        """
        return await dispatch("plugin_list", lambda: service.plugin_list(registry=registry))

    @mcp.tool
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

    @mcp.tool
    async def plugin_remove(name: str) -> dict[str, Any]:
        """Disable a plugin. The distribution stays installed and is not touched.

        Args:
            name: The plugin's name, as ``plugin_list`` reports it.
        """
        return await dispatch("plugin_remove", lambda: service.plugin_remove(name))

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


__all__ = ["INSTRUCTIONS", "SERVER_NAME", "TOOL_NAMES", "build_server"]
