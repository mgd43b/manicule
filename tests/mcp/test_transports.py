"""Which tools a transport carries, decided in one place and asserted from both ends.

The rule is one sentence: **an authenticated socket carries the read-only tools and one named
write, an unauthenticated one carries the reads alone, and a pipe carries all of them.** It is
not a setting and there is no caller entitled to a different answer — see
:data:`~manicule.mcp.serve.NETWORK_SURFACE_IS_READ_ONLY` for the transport rule and
:func:`~manicule.mcp.server.network_authoring` for the one exception, which is a frozenset
conditioned on authentication rather than a flag any registration can set for itself.

What is here is the rule *at the transport*. Two things it deliberately is not:
``tests/api/test_routes.py`` asserts the absences by name against the mounted endpoint a client
actually reaches, and ``tests/mcp/test_stdio.py`` asserts the whole surface over a real pipe in a
real second process. This file is the seam between them, and it exists because the seam is where
the two could come apart: a change that filtered the wrong transport would leave one of those
green.
"""

from __future__ import annotations

import pytest

from manicule.app.results import Check
from manicule.config.settings import AuthMode, AuthoringSettings, AuthSettings, Settings
from manicule.core.errors import PolicyError
from manicule.mcp.serve import NETWORK_SURFACE_IS_READ_ONLY, address_for, surface
from manicule.mcp.server import NETWORK_AUTHORING, TOOL_NAMES, network_authoring
from tests.app.fakes import FakeBackend
from tests.mcp.test_annotations import MUTATIONS

from manicule.app.service import ApplicationService  # isort: skip


EVERYWHERE = "0.0.0.0"  # noqa: S104 - the address whose exposure is the subject


@pytest.fixture
def service() -> ApplicationService:
    """The default installation, which has **no authentication** — ``auth.mode`` defaults to none.

    That was scenery before and is a precondition now: ``network_authoring`` is empty without
    authentication, so this service's socket carries the reads and nothing else.
    """
    return ApplicationService(FakeBackend())


@pytest.fixture
def authenticated() -> ApplicationService:
    """The installation whose socket carries its one write."""
    base = Settings()
    security = base.security.model_copy(update={"auth": AuthSettings(mode=AuthMode.API_KEY)})
    return ApplicationService(FakeBackend(settings=base.model_copy(update={"security": security})))


def test_stdio_carries_every_tool(service: ApplicationService) -> None:
    """A pipe between one client and one process has no network to be reached from.

    So the classification buys nothing here, and applying it would take the write tools away from
    the deployment they exist for: an editor spawning ``manicule serve``.
    """
    assert sorted(surface(service, transport="stdio").tools) == sorted(TOOL_NAMES)


def test_a_socket_carries_the_read_only_tools_and_one_named_write(
    authenticated: ApplicationService,
) -> None:
    """The rule, from the transport's side, against the classification's own list.

    ``MUTATIONS`` is imported from the annotations suite rather than restated, because those two
    lists being the same list is the point: what a tool says it does decides whether a socket may
    carry it, and a second list here would be a second answer.

    ``NETWORK_AUTHORING`` is the one thing that is *not* derived from what a tool says it does,
    and it is subtracted explicitly here rather than folded into ``MUTATIONS`` for that reason.
    Authoring writes, this file says so, and it is on a socket anyway — by name, in one constant,
    with :func:`~manicule.app.bind.require_authoring_authentication` guarding what a socket
    carrying it must have. Every other mutating tool stays absent, which is what the first
    assertion is.

    **This socket is authenticated, and that is now a precondition rather than scenery.** The
    test below is the same seam with authentication off.
    """
    carried = set(surface(authenticated, transport="http").tools)

    assert carried & (set(MUTATIONS) - NETWORK_AUTHORING) == set(), sorted(
        carried & (set(MUTATIONS) - NETWORK_AUTHORING)
    )
    assert carried | set(MUTATIONS) == set(TOOL_NAMES)
    assert carried >= NETWORK_AUTHORING, "the named write is not on the surface it is named for"


def test_an_unauthenticated_socket_carries_no_write_at_all(service: ApplicationService) -> None:
    """The seam with authentication off: the reads, and not one thing that writes.

    **Asserted as a set operation**, the way the network surface is asserted everywhere else, so
    a write let onto an unauthenticated socket by any route fails here rather than needing to be
    named first. The same rule from the other side of the same seam as
    ``tests/api/test_routes.py``'s pair of set-operation tests.

    The reason is not tidiness. With no authentication there is no credential, so an anonymous
    caller resolves to an administrator and ``require_network_member``'s member floor is cleared
    by everybody — a ``document_create`` published here would be callable by anything that can
    route to the port, writing into a corpus assistants read back as standing instructions.

    **Independent of the bind**, deliberately: this is the surface on loopback too, so
    ``--no-authentication`` cannot widen it, and an unauthenticated local port carries no write
    either. ``network_authoring`` is asserted alongside the surface so that a future change
    making the *constant* empty — which would pass the first assertion — is told apart from this
    one.
    """
    assert service.settings.security.auth.mode is AuthMode.NONE
    carried = set(surface(service, transport="http").tools)

    assert network_authoring(service.settings) == frozenset()
    assert frozenset({"document_create"}) == NETWORK_AUTHORING, (
        "the constant changed rather than the condition on it, so this test would pass against a "
        "surface that had simply lost its one write everywhere"
    )
    assert carried & set(MUTATIONS) == set(), sorted(carried & set(MUTATIONS))
    assert carried | set(MUTATIONS) == set(TOOL_NAMES)


def test_a_socket_serving_authoring_without_authentication_refuses_to_start(
    service: ApplicationService,
) -> None:
    """Refused before a socket exists, rather than served and declined per call.

    The refusal is what makes ``NETWORK_AUTHORING`` acceptable on a **loopback** bind, which
    ``resolve_bind`` waves through without any of the three things a wide bind needs — so
    without this, an unauthenticated local port would carry a tool that writes into a corpus
    every assistant on the machine recalls as instructions.

    Asserted through ``address_for``, which is what the MCP-only transport calls before binding,
    so this is the path a real start takes rather than a helper written for a test. The
    application everything else is served from refuses in ``manicule.api.app.build_app``, which
    ``tests/api/test_routes.py`` drives.
    """
    settings = service.settings.model_copy(
        update={
            "authoring": AuthoringSettings(source="memories", collections=("memory",)),
            "security": service.settings.security.model_copy(
                update={"auth": AuthSettings(mode=AuthMode.NONE)}
            ),
        }
    )
    configured = ApplicationService(FakeBackend(settings=settings))

    with pytest.raises(PolicyError, match="authoring"):
        address_for(configured, transport="http")

    # And the same configuration over a pipe starts, because a pipe has no port. Asserted in the
    # same test so that a refusal widened to every transport fails here rather than being
    # discovered by somebody whose editor stopped being able to spawn manicule.
    assert address_for(configured, transport="stdio").transport == "stdio"


async def test_deciding_a_socket_address_records_it_and_a_pipe_records_nothing(
    service: ApplicationService,
) -> None:
    """The MCP half of the same rule, and the stdio case that must stay silent.

    This transport publishes the ``doctor`` tool, so a client asking it about a server started
    with ``--host 0.0.0.0`` would otherwise be told the process is reachable only from the
    machine it is on.

    The pipe is asserted in the same test because it is the way this goes wrong: a
    ``ServerAddress`` for stdio carries ``host=""``, which is in ``EVERY_INTERFACE`` rather than
    ``LOOPBACK_HOSTS``, so a recording placed above the stdio branch would diagnose a process
    with no socket at all as a wide bind — reporting the loudest possible finding about the
    quietest possible transport.
    """
    address_for(
        service, transport="http", host=EVERYWHERE, allow_public=True, allow_unauthenticated=True
    )
    assert (await _transport_check(service)).facts["bind_host"] == EVERYWHERE

    piped = ApplicationService(FakeBackend())
    address_for(piped, transport="stdio")
    check = await _transport_check(piped)
    assert check.state == "ok"
    assert check.facts["bind_host"] == "127.0.0.1"
    assert check.facts["loopback"] is True


async def _transport_check(service: ApplicationService) -> Check:
    """The ``transport`` check, through ``doctor`` — the operation every surface actually calls."""
    diagnosis = await service.doctor()
    return next(check for check in diagnosis.checks if check.name == "transport")


def test_a_socket_without_authoring_configured_still_starts_unauthenticated(
    service: ApplicationService,
) -> None:
    """The condition is authoring being *configured*, not the tool existing.

    Without this, the check above would make every read-only loopback socket require an API key
    — a change to how manicule is served, imposed by a feature that installation does not use.
    The socket that starts carries no write either way: this one is unauthenticated, so
    ``network_authoring`` is empty and ``document_create`` is not registered on it.
    """
    assert service.settings.authoring.configured is False
    assert address_for(service, transport="http").transport == "http"
    assert "document_create" not in surface(service, transport="http").tools


def test_the_authoring_refusal_is_not_waived_by_the_no_authentication_flag(
    service: ApplicationService,
) -> None:
    """**This is where the escape hatch stops, and it stops on purpose.**

    ``--no-authentication`` satisfies ``resolve_bind``'s third condition, which is a statement
    about *reading* an index: an operator may decide their own corpus is readable by their own
    network. This refusal is a statement about *writing* into a corpus that assistants read back
    as standing instructions, and no argument makes an anonymous caller safe to hand that to.

    **Waiving it would also have closed only one door.** Emptying the MCP surface takes
    ``document_create`` off the socket; the same operation is `POST /api/v1/documents` on the
    HTTP surface, asking for a member floor that an anonymous administrator clears. A flag that
    let this configuration start would therefore have opened a write path that the surface
    narrowing does not reach — which is exactly the shape of hole that looks closed.

    So an installation that wants authoring served over a network wants an API key, and the
    refusal says so by name.
    """
    settings = service.settings.model_copy(
        update={"authoring": AuthoringSettings(source="memories", collections=("memory",))}
    )
    configured = ApplicationService(FakeBackend(settings=settings))
    assert configured.settings.security.auth.mode is AuthMode.NONE

    with pytest.raises(PolicyError, match="authoring") as caught:
        address_for(configured, transport="http", allow_unauthenticated=True)
    assert "--no-authentication does not waive this" in str(caught.value)

    # And a pipe still carries it, because a pipe has no port. Same test, so a refusal widened
    # to every transport fails here rather than being found by somebody whose editor stopped
    # being able to spawn manicule.
    assert address_for(configured, transport="stdio").transport == "stdio"
    assert "document_create" in surface(configured, transport="stdio").tools


def test_the_announced_tool_count_is_what_the_transport_offers(
    service: ApplicationService,
) -> None:
    """The line an operator reads at startup agrees with what ``tools/list`` will say.

    It did not have to: the count was ``len(TOOL_NAMES)`` for both transports, which would now
    announce forty-five on a socket that carries twenty-seven — the banner disagreeing with the
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
