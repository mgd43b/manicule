"""What a route is given: the application service, and the proxy policy.

Both hang off the application rather than being constructed per request. The service owns a
runtime with a database engine and — once something asks — a model runtime, so building one per
request would rebuild those; the proxy policy is a parsed form of configuration and changing it
means restarting anyway.

They are read through dependencies rather than module globals so that a suite can build an
application over a fake backend and drive the real routing, which is what makes the parity and
tenancy suites here worth anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from manicule.api.proxy import ProxyPolicy
    from manicule.app.service import ApplicationService


def service_of(request: Request) -> ApplicationService:
    """The one service this application serves.

    Raises:
        ManiculeError: The application was assembled without one. A defect rather than a
            request problem, and it propagates as one.
    """
    service: ApplicationService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - build_app always sets it
        msg = "the application was built without an application service"
        raise ManiculeError(msg)
    return service


def policy_of(request: Request) -> ProxyPolicy:
    """The trusted-proxy policy this application was built with."""
    policy: ProxyPolicy | None = getattr(request.app.state, "proxy_policy", None)
    if policy is None:  # pragma: no cover - build_app always sets it
        msg = "the application was built without a proxy policy"
        raise ManiculeError(msg)
    return policy


Service = Annotated["ApplicationService", Depends(service_of)]
Policy = Annotated["ProxyPolicy", Depends(policy_of)]


__all__ = ["Policy", "Service", "policy_of", "service_of"]
