"""The read-only path, over a real pipe, in a real second process.

Every other suite here drives the server in memory, which exercises the tool registry and the
envelope and nothing about the transport. This one spawns ``python -m tests.mcp.stdio_child``
and speaks MCP to it over stdin and stdout, because the deployment everybody actually uses is a
client spawning a process — and the failure that deployment has is a byte on stdout that is not
a message.

What it establishes is narrow and deliberate: a client can discover the surface, resolve a
collection, get its counts and run a scoped search, and **nothing was indexed, synchronized or
deleted to make that work**. The corpus is the same synthetic fixture the rest of this package
uses; see ``tests/mcp/stdio_child.py`` for what is real in the child and what is not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from tests.mcp.qualification import CHUNK_COUNT, COLLECTION, DOCUMENT_COUNT

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]

WRITE_TOOLS = frozenset(
    {
        "index_path",
        "connector_sync",
        "document_delete",
        "document_reindex",
        "collection_create",
        "collection_delete",
        "config_set",
    }
)
"""Tools this test must not call. Named so the claim in the docstring is checkable below."""


def _transport() -> StdioTransport:
    return StdioTransport(
        command=sys.executable, args=["-m", "tests.mcp.stdio_child"], cwd=str(REPO_ROOT)
    )


async def test_a_stdio_client_can_list_count_and_search_without_changing_anything() -> None:
    """One session, four calls, and a scoped search that returns real passages.

    The tool names are recorded as they are called and checked against
    :data:`WRITE_TOOLS` at the end, so "no corpus mutation" is asserted rather than asserted
    about — a later edit that added an ``index_path`` call to make the search return something
    would fail here instead of quietly changing what this test proves.
    """
    called: list[str] = []
    async with Client(_transport()) as client:
        published = {tool.name for tool in await client.list_tools()}
        assert "search" in published
        assert "collection_counts" in published

        async def payload(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            called.append(name)
            result = await client.call_tool(name, arguments)
            envelope: dict[str, Any] = dict(result.structured_content or {})
            assert envelope["ok"] is True, envelope
            data: dict[str, Any] = envelope["data"]
            return data

        listed = await payload("collection_list", {})
        collections: Sequence[dict[str, Any]] = listed["collections"]
        assert len(collections) == 1
        target = collections[0]
        assert target["name"] == COLLECTION

        counted = await payload("collection_counts", {"collection_id": target["id"]})
        assert counted["documents"] == DOCUMENT_COUNT
        assert counted["chunks"] == CHUNK_COUNT

        found = await payload(
            "search",
            {
                "query": "which component owns admission control",
                "collections": [COLLECTION],
                "limit": 4,
            },
        )
        assert found["collections"] == [COLLECTION]
        assert found["hits"], "the scoped search returned nothing, so it proves nothing"

    assert set(called) & WRITE_TOOLS == set()


async def test_the_read_only_annotations_survive_the_transport() -> None:
    """The hints are on ``tools/list`` over the wire, not only on the in-memory object.

    Worth its own assertion because a client's approval decision is made from what arrives, and
    an annotation lost in serialization would leave every in-memory test in this package green
    while the reported problem — a client gating ``search`` — stayed exactly as it was.
    """
    async with Client(_transport()) as client:
        published = {tool.name: tool.annotations for tool in await client.list_tools()}

    for name in ("collection_list", "collection_counts", "collection_documents", "search"):
        annotations = published[name]
        assert annotations is not None, f"{name} arrived with no annotations"
        assert annotations.readOnlyHint is True, name
        assert annotations.destructiveHint is False, name
        assert annotations.idempotentHint is True, name
        assert annotations.openWorldHint is False, name

    assert published["document_delete"] is not None
    assert published["document_delete"].readOnlyHint is False
