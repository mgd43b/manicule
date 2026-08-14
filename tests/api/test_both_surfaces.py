"""One process, both surfaces, more than one client, over a real socket.

Every assertion here is about the *transport*, which is why this suite pays for a port while
the rest of ``tests/api`` does not. Two clients are two connections; a wedged client is a
connection whose request is not finishing; a disconnect is a socket closing. Starlette's test
client dispatches into the application inside the caller's own task, so none of those exist
there and every one of these tests would pass whatever the answer was.

**What is shared, and what is not.** One :class:`~manicule.app.service.ApplicationService` over
one backend answers every call on both surfaces, so the ``Runtime``, the pipeline, the session
vault and the scheduler are one instance each — deliberately, because they are facts about the
*process*: it holds one writer lock, one set of sessions and one schedule. What is per client is
nothing at all. The mount is stateless (``manicule.api.app.build_app``), so a call carries its
own arguments, is answered, and leaves nothing behind that a second client could observe or a
first could rely on.

That is asserted rather than asserted-about: :func:`test_one_client_s_scope_does_not_reach_another`
gives two clients different arguments at the same time and checks each got its own answer, which
is what "no shared session" means in the only terms a caller can see.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from fastmcp import Client

from tests.api.live import serving
from tests.api.support import backend_with_a_document
from tests.app.fakes import FakeRetriever
from tests.ingest.fakes import Gate

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from manicule.core.retrieval import Query
    from manicule.retrieval.retriever import RetrievalResult

OK = 200
NOT_FOUND = 404

COLLECTION = "Engineering Architecture"
"""A collection to scope a search to. Synthetic, like every name in these suites."""

TIMEOUT_S = 30.0
"""How long any wait here is given before the test fails.

A bound rather than a duration anything is measured against. Every assertion below is an
*arrival* — a call returned, two calls were inside at once — for the reason
``tests/ingest/test_concurrency.py`` states: a test that waited a fixed time and then looked
would pass against a sequential server whenever the wait happened to be long enough.
"""


@dataclass
class Parking(FakeRetriever):
    """A retriever that parks inside ``retrieve`` until a test lets it out.

    The only way to have a call genuinely in flight on the server while the test does something
    else. Without it "two clients at once" is two calls that took turns quickly, which is a
    statement about nothing.

    Built on ``tests.ingest.fakes.Gate`` rather than on an event of its own, because the gate
    already answers the question these tests ask — :meth:`~tests.ingest.fakes.Gate.wait_for`
    returns when the stated number of callers are inside **at once**, and fails saying how many
    ever were. An event would answer "at least one arrived", which is a weaker claim than any
    test here is making.
    """

    gate: Gate = field(default_factory=Gate)
    finished: int = 0
    """How many calls came out the far side.

    Incremented after the parent has answered, so it counts *completions* — which is what every
    "and it had not finished yet" assertion below needs. ``FakeRetriever.seen`` counts arrivals
    past the gate and is a different number.
    """

    completed: asyncio.Event = field(default_factory=asyncio.Event)
    """Set as each call finishes, so a test can await a completion rather than poll for one.

    Needed by exactly one test — the one whose client is gone, so there is no answer to await and
    the only observable is the server having carried on. Everywhere else the call itself is
    awaited and this is redundant.
    """

    @override
    async def retrieve(self, query: Query) -> RetrievalResult:
        await self.gate.pass_through()
        result = await super().retrieve(query)
        self.finished += 1
        self.completed.set()
        return result


def _parking(backend: Any) -> Parking:
    """Swap the backend's retriever for one that parks, keeping the candidates it had."""
    parking = Parking(candidates=list(backend.retriever_.candidates))
    backend.retriever_ = parking
    return parking


async def test_both_surfaces_answer_from_one_process_at_the_same_time() -> None:
    """An HTTP request and an MCP call, concurrently, against one server.

    Concurrently rather than one after the other, and the ordering is the assertion: the MCP
    call is held inside the retriever while the HTTP request is made and answered. A server that
    served the two surfaces by taking turns — one event loop blocked by the other, one lock
    around the service — fails here and passes a sequential version of the same test.
    """
    backend, _ = backend_with_a_document()
    parking = _parking(backend)

    async with serving(backend) as live, live.mcp() as mcp, live.http() as http:
        search = asyncio.create_task(mcp.call_tool("search", {"query": "retry", "limit": 1}))
        await parking.gate.wait_for(1)

        # The MCP call is inside the retriever and has not returned. The API answers anyway.
        answered = await http.get("/api/v1/stats")
        assert answered.status_code == OK, answered.text
        assert answered.json()["ok"] is True
        assert parking.finished == 0, "the MCP call finished, so the two never overlapped"

        parking.gate.open()
        envelope = (await asyncio.wait_for(search, timeout=TIMEOUT_S)).structured_content or {}

    assert envelope["ok"] is True, envelope
    assert envelope["op"] == "search"


async def test_two_mcp_clients_are_connected_at_once_and_each_gets_its_own_answer() -> None:
    """Two clients, two calls in flight together, each answered with what *it* asked for.

    The two queries differ, and both are inside the retriever at the same time before either is
    released. That is what makes the answers a statement about isolation rather than about
    ordering: a server holding one client's arguments in shared state would answer the second
    call with the first's scope, and a server that serialized them would never have both inside.
    """
    backend, _ = backend_with_a_document()
    parking = _parking(backend)

    async with serving(backend) as live, live.mcp() as first, live.mcp() as second:
        one = asyncio.create_task(first.call_tool("search", {"query": "alpha", "limit": 1}))
        two = asyncio.create_task(second.call_tool("search", {"query": "beta", "limit": 1}))
        # Both inside at once, neither answered. `wait_for` is what makes that a fact rather than
        # a hope: it returns when two callers are in the gate **together**, and fails saying how
        # many ever were if they never are — which is what a server serving one client at a time
        # would look like from here.
        await parking.gate.wait_for(2)
        assert parking.finished == 0
        assert parking.gate.peak >= 2, "the two calls took turns rather than overlapping"

        parking.gate.open()
        answers = await asyncio.wait_for(asyncio.gather(one, two), timeout=TIMEOUT_S)

    scopes = [(answer.structured_content or {})["data"]["query"] for answer in answers]
    assert scopes == ["alpha", "beta"], (
        f"each client's answer must carry its own query; got {scopes}"
    )


async def test_one_client_s_scope_does_not_reach_another() -> None:
    """The same property stated over the argument a leak would be worst in: the collection scope.

    A scope that survived one call and reached the next is the failure ``manicule.mcp.server``'s
    instructions warn a client about in as many words — "nothing is remembered between calls" —
    and the shape it would take on a served surface is one client's scope answering another's
    question. Asserted through ``data.collections``, which is the field the surface publishes
    precisely so a caller can check the scope arrived.
    """
    backend, _ = backend_with_a_document()
    # A real collection, because a name this workspace does not have is refused rather than
    # searched — which would make the scoped call fail and the assertion below pass for the
    # wrong reason.
    await backend.organization_.create_collection(COLLECTION)

    async with serving(backend) as live, live.mcp() as first, live.mcp() as second:
        scoped = await first.call_tool(
            "search", {"query": "retry", "collections": [COLLECTION], "limit": 1}
        )
        unscoped = await second.call_tool("search", {"query": "retry", "limit": 1})

    first_envelope = scoped.structured_content or {}
    assert first_envelope["ok"] is True, first_envelope
    assert first_envelope["data"]["collections"] == [COLLECTION]
    assert (unscoped.structured_content or {})["data"]["collections"] == [], (
        "the second client inherited the first's collection scope"
    )


async def test_a_wedged_client_does_not_stall_another() -> None:
    """One client's call is stuck in the retriever; the other's is answered while it is stuck.

    "Wedged" here is the shape that actually happens: a call the server cannot finish, because
    the thing it is waiting on has not happened. A second client asking for something the server
    *can* answer gets its answer, which is the property — a surface serializing every call behind
    one lock, or serving one connection at a time, would make the second wait for the first.
    """
    backend, _ = backend_with_a_document()
    parking = _parking(backend)

    async with serving(backend) as live, live.mcp() as wedged, live.mcp() as other:
        stuck = asyncio.create_task(wedged.call_tool("search", {"query": "stuck", "limit": 1}))
        await parking.gate.wait_for(1)

        # `stats` reaches the store rather than the retriever, so it is answerable while the
        # other call is not. A tool that also parked would be testing the fake, not the server.
        answered = await asyncio.wait_for(other.call_tool("stats", {}), timeout=TIMEOUT_S)
        assert (answered.structured_content or {})["ok"] is True
        assert parking.finished == 0, "the wedged call completed, so nothing was ever wedged"

        parking.gate.open()
        await asyncio.wait_for(stuck, timeout=TIMEOUT_S)


async def test_a_client_that_disconnects_does_not_cancel_the_work_it_started() -> None:
    """The server is the writer; a client detaching is not a stop signal.

    #139 established this for the control socket, where a proxied sync outlives the terminal that
    asked for it. The same has to hold here, and for a stronger reason than symmetry: an editor
    restarting its MCP client mid-call is ordinary, and a call that unwound halfway would leave
    whatever it was doing half done with nothing reporting it.

    The call is abandoned as rudely as a client can manage — the request task canceled and the
    transport closed under it — and then the thing it was waiting on is released. ``finished``
    going to one is the server having carried on regardless.
    """
    backend, _ = backend_with_a_document()
    parking = _parking(backend)

    async with serving(backend) as live:
        client = live.mcp()
        await client.__aenter__()
        call = asyncio.create_task(client.call_tool("search", {"query": "abandoned", "limit": 1}))
        await parking.gate.wait_for(1)

        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await asyncio.gather(_closed(client), return_exceptions=True)
        assert parking.finished == 0, "the call finished before the client was even gone"

        parking.gate.open()
        await asyncio.wait_for(parking.completed.wait(), timeout=TIMEOUT_S)

    assert parking.finished == 1, "the server abandoned the call when its client went away"


async def test_the_browser_surface_and_mcp_are_served_from_the_same_port() -> None:
    """One address, three things behind it, which is the whole point of the arrangement.

    Asserted over the same running server rather than over three applications, because "they can
    each be built" was already true and is not what an operator needs — what they need is one
    port in one plist and one line in an MCP client's configuration.
    """
    backend, _ = backend_with_a_document()

    async with serving(backend) as live, live.http() as http, live.mcp() as mcp:
        page = await http.get("/ui")
        api = await http.get("/api/v1/documents")
        tools = await mcp.list_tools()

    assert page.status_code == OK, "the browser surface is not on this port"
    assert api.status_code == OK, "the JSON API is not on this port"
    assert tools, "MCP is on this port and offered nothing"


async def test_no_web_removes_the_browser_surface_and_leaves_mcp() -> None:
    """``--no-web`` reduces what a process exposes; it does not reduce something it did not name.

    The mirror of ``tests/app/test_serving.py``'s ``--no-web`` suite, extended to the surface
    that arrived after it. A flag that took MCP with it would be a flag switching off more than
    it names, which is that suite's other assertion — made there against the JSON API and here
    against the endpoint an editor is configured to reach.
    """
    backend, _ = backend_with_a_document()

    async with serving(backend, web=False) as live, live.http() as http, live.mcp() as mcp:
        page = await http.get("/ui")
        tools = await mcp.list_tools()

    assert page.status_code == NOT_FOUND, "the browser surface was served with --no-web"
    assert tools, "--no-web took the MCP surface with it"


async def _closed(client: Client[Any]) -> None:
    """Close a client that was opened by hand, for the one test that cannot use ``async with``.

    That test needs the client to go away *while a call is in flight*, which a context manager
    cannot express: the block would have to end inside itself. Wrapped in a function so the
    exit is named once rather than at the call site.
    """
    closing: AbstractAsyncContextManager[object] = client
    await closing.__aexit__(None, None, None)
