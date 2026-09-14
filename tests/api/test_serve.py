"""Deciding where the API listens, without opening a socket.

The decision is separated from the listening precisely so it can be asserted here: a bind that
is only decided inside the call that performs it is a bind nobody can test, and the refusals
are the point.

The policy itself is `manicule.app.bind.resolve_bind` and is covered by ``tests/app/test_bind.py``.
What is checked here is that this surface **goes through it** and hands the decision on — so
that ``build_app``'s own refusal sees the address that was decided rather than the one
configuration happens to hold.
"""

from __future__ import annotations

import pytest

from manicule.api.app import ROUTE_GROUPS
from manicule.api.serve import TRANSPORT, address_for, application
from manicule.app.service import ApplicationService
from manicule.core.errors import PolicyError
from tests.api.support import backend_with_a_document

WIDE = "192.0.2.10"
"""TEST-NET-1: routable-looking and nobody's."""


def test_the_default_address_is_loopback() -> None:
    """No configuration, no flags."""
    backend, _ = backend_with_a_document()
    bind, address = address_for(ApplicationService(backend))
    assert bind.loopback
    assert address.host == "127.0.0.1"
    assert address.loopback
    assert address.transport == TRANSPORT


def test_the_address_reports_the_route_groups_rather_than_a_tool_count() -> None:
    """The field counts what the surface offers, and this surface offers groups.

    Twelve since MCP was mounted on this application: it is a group this surface offers, so it
    is counted like the other eleven. The literal is kept beside the length rather than replaced
    by it, because ``len(ROUTE_GROUPS) == len(ROUTE_GROUPS)`` would pass against an empty tuple.
    """
    backend, _ = backend_with_a_document()
    _, address = address_for(ApplicationService(backend))
    assert address.tools == len(ROUTE_GROUPS) == 12


def test_a_wide_host_is_refused_without_the_explicit_flag() -> None:
    """Configuration alone cannot widen the bind, on this surface as on every other."""
    backend, _ = backend_with_a_document(
        security={"transport": {"bind_host": WIDE}, "auth": {"mode": "api_key"}}
    )
    with pytest.raises(PolicyError, match="--allow-public-bind"):
        address_for(ApplicationService(backend))


def test_a_wide_host_is_refused_without_authentication() -> None:
    """The flag on its own is not enough."""
    backend, _ = backend_with_a_document(security={"transport": {"bind_host": WIDE}})
    with pytest.raises(PolicyError, match=r"security\.auth\.mode"):
        address_for(ApplicationService(backend), allow_public=True)


def test_a_wide_bind_is_possible_when_all_three_conditions_hold() -> None:
    """The positive control. A bind decision that can only refuse is not a decision."""
    backend, _ = backend_with_a_document(
        security={"transport": {"bind_host": WIDE}, "auth": {"mode": "api_key"}}
    )
    bind, address = address_for(ApplicationService(backend), allow_public=True)
    assert not bind.loopback
    assert address.host == WIDE
    assert not address.loopback


def test_building_the_application_refuses_a_wide_bind_that_was_asked_for_without_auth() -> None:
    """``application`` resolves the bind **first** and hands it to ``build_app``.

    Without that ordering the second refusal would read configuration rather than the decided
    address, and a command line that named a wide host would slip past it.
    """
    backend, _ = backend_with_a_document()
    with pytest.raises(PolicyError):
        application(ApplicationService(backend), host=WIDE, allow_public=True)


def test_both_flags_build_a_wide_unauthenticated_application() -> None:
    """The escape hatch has to clear **both** refusals, and this is where that is proved.

    ``application`` passes ``allow_unauthenticated`` to the bind *and* to ``build_app``, because
    they refuse separately and a flag that satisfied only the first would be a flag that appears
    to work and then fails one layer down — with a message about a decision the operator has
    already made. The previous test is the same call without the flag, so the pair is what says
    the argument is doing the work rather than the configuration.
    """
    backend, _ = backend_with_a_document()
    app, address = application(
        ApplicationService(backend), host=WIDE, allow_public=True, allow_unauthenticated=True
    )
    assert address.host == WIDE
    assert not address.loopback
    assert app.title == "manicule"


async def _transport_facts(service: ApplicationService) -> dict[str, object]:
    """The ``transport`` check's facts, through ``doctor`` rather than the private method.

    What is under test is the diagnosis a caller receives — over the API health route, the MCP
    tool, or the command line — so the assertion goes through the operation all three call.
    """
    diagnosis = await service.doctor()
    return dict(next(check for check in diagnosis.checks if check.name == "transport").facts)


async def test_deciding_an_address_records_it_for_the_diagnosis() -> None:
    """Every serving path reaches a bind through ``address_for``, so recording lives there.

    **Not in the command line.** This entry point is public: an embedder calls
    ``manicule.api.serve.serve``, a production ASGI server builds the application itself, and
    both expose ``doctor`` at ``GET /api/v1/health``. A recording wired into ``manicule serve``
    would leave every one of those describing ``security.transport.bind_host`` — which a
    caller-supplied ``host=`` never touches — so the health route of a server answering
    ``0.0.0.0`` would report "reachable only from this machine".

    Asserted through the ``transport`` check rather than on the attribute, because what is under
    test is the diagnosis a caller receives.
    """
    backend, _ = backend_with_a_document()
    service = ApplicationService(backend)

    address_for(service, host=WIDE, allow_public=True, allow_unauthenticated=True)

    facts = await _transport_facts(service)
    assert facts["bind_host"] == WIDE
    assert facts["loopback"] is False
    assert facts["configured_bind_host"] == "127.0.0.1"


async def test_a_refused_address_records_nothing() -> None:
    """A bind that was never decided is not a bind, and must not be diagnosed as one.

    Without this the recording could be moved above ``resolve_bind`` and every refusal test
    would still pass, while a process that refused to start described an exposure it never had.
    """
    backend, _ = backend_with_a_document()
    service = ApplicationService(backend)

    with pytest.raises(PolicyError):
        address_for(service, host=WIDE, allow_public=True)

    assert (await _transport_facts(service))["bind_host"] == "127.0.0.1"


def test_a_command_line_host_that_is_loopback_builds() -> None:
    """The other direction: a decided loopback address is allowed even with no auth."""
    backend, _ = backend_with_a_document()
    app, address = application(ApplicationService(backend), host="127.0.0.1", port=9999)
    assert address.port == 9999
    assert address.loopback
    assert app.title == "manicule"


def test_an_out_of_range_port_is_refused_before_anything_is_built() -> None:
    backend, _ = backend_with_a_document()
    with pytest.raises(PolicyError):
        address_for(ApplicationService(backend), port=0)


def test_the_transport_name_distinguishes_the_api_from_the_mcp_server() -> None:
    """Both speak HTTP. An operator reading a pid file should not have to guess which."""
    from manicule.mcp.serve import address_for as mcp_address_for  # noqa: PLC0415

    backend, _ = backend_with_a_document()
    service = ApplicationService(backend)
    _, api = address_for(service)
    mcp = mcp_address_for(service, transport="http")
    assert api.transport != mcp.transport
    assert api.transport == "http-api"
