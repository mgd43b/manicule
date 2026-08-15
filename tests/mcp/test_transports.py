"""Which tools a transport carries, decided in one place and asserted from both ends.

The rule is one sentence: **a socket carries the read-only tools; a pipe carries all of them.**
It is not a setting and there is no caller entitled to the other answer — see
:data:`~manicule.mcp.serve.NETWORK_SURFACE_IS_READ_ONLY`.

What is here is the rule *at the transport*. Two things it deliberately is not:
``tests/api/test_routes.py`` asserts the absences by name against the mounted endpoint a client
actually reaches, and ``tests/mcp/test_stdio.py`` asserts the whole surface over a real pipe in a
real second process. This file is the seam between them, and it exists because the seam is where
the two could come apart: a change that filtered the wrong transport would leave one of those
green.
"""

from __future__ import annotations

import pytest

from manicule.mcp.serve import NETWORK_SURFACE_IS_READ_ONLY, address_for, surface
from manicule.mcp.server import TOOL_NAMES
from tests.app.fakes import FakeBackend
from tests.mcp.test_annotations import MUTATIONS

from manicule.app.service import ApplicationService  # isort: skip


@pytest.fixture
def service() -> ApplicationService:
    return ApplicationService(FakeBackend())


def test_stdio_carries_every_tool(service: ApplicationService) -> None:
    """A pipe between one client and one process has no network to be reached from.

    So the classification buys nothing here, and applying it would take the write tools away from
    the deployment they exist for: an editor spawning ``manicule serve``.
    """
    assert sorted(surface(service, transport="stdio").tools) == sorted(TOOL_NAMES)


def test_a_socket_carries_the_read_only_tools_and_no_others(
    service: ApplicationService,
) -> None:
    """The rule, from the transport's side, against the classification's own list.

    ``MUTATIONS`` is imported from the annotations suite rather than restated, because those two
    lists being the same list is the point: what a tool says it does decides whether a socket may
    carry it, and a second list here would be a second answer.
    """
    carried = set(surface(service, transport="http").tools)

    assert carried & set(MUTATIONS) == set(), sorted(carried & set(MUTATIONS))
    assert carried | set(MUTATIONS) == set(TOOL_NAMES)


def test_the_announced_tool_count_is_what_the_transport_offers(
    service: ApplicationService,
) -> None:
    """The line an operator reads at startup agrees with what ``tools/list`` will say.

    It did not have to: the count was ``len(TOOL_NAMES)`` for both transports, which would now
    announce twenty-nine on a socket that carries fourteen — the banner disagreeing with the
    protocol on the one number somebody would check against their client.
    """
    over_a_pipe = address_for(service, transport="stdio")
    over_a_socket = address_for(service, transport="http")

    assert over_a_pipe.tools == len(TOOL_NAMES)
    assert over_a_socket.tools == len(surface(service, transport="http").tools)
    assert over_a_socket.tools < over_a_pipe.tools, (
        "the socket announced as many tools as the pipe, so either the filter did not run or "
        "every tool is now read-only"
    )


def test_there_is_no_way_to_ask_for_the_write_tools_on_a_socket() -> None:
    """The absence of a switch, asserted because an absence has no behavior to observe.

    A setting that turned this off would trade a structural guarantee for a configuration one,
    which is a weaker guarantee that fails silently — the surface would look identical and the
    tools would be there. The constant is read by exactly one expression, in
    :func:`~manicule.mcp.serve.surface`, and nothing loads it from configuration.
    """
    import ast  # noqa: PLC0415 - only this assertion parses a module
    from pathlib import Path  # noqa: PLC0415

    import manicule.mcp.serve as transports  # noqa: PLC0415 - located rather than imported

    assert NETWORK_SURFACE_IS_READ_ONLY is True
    tree = ast.parse(Path(str(transports.__file__)).read_text(encoding="utf-8"))

    # Parsed rather than searched as text, so a mention in a docstring or in `__all__` is not
    # counted as a read. What is being asserted is that exactly one *expression* consults the
    # constant, which is the one in `surface` that decides what a transport carries.
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "NETWORK_SURFACE_IS_READ_ONLY"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(reads) == 1, (
        f"the constant is read {len(reads)} times, so the surface is decided in more than one "
        f"place and the two can disagree"
    )

    consulted = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"getenv", "environ"}
    }
    assert consulted == set(), (
        f"this module reads {sorted(consulted)}, so what a socket carries could be decided by "
        f"something other than the transport"
    )
