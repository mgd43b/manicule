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
