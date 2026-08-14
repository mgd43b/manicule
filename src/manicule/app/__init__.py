"""The application service layer, which is the only place manicule's behavior lives.

The command line and the MCP server are both adapters over :class:`ApplicationService`. So is
the HTTP API when it arrives. Neither surface decides anything: they parse arguments, call one
method, and render what comes back.

That is not an aesthetic preference. Every rule that lives in a surface is a rule the other
surfaces do not have, and the MCP tool is the one called unattended — so a workspace check
implemented in the CLI is a workspace check an assistant can walk around. Putting the rules
here makes "both surfaces behave identically" a property a test can fail on, and
``tests/app/test_surface_parity.py`` is that test.

Importing this package pulls in no database, no model runtime and no web framework. The
service is written against the protocols in :mod:`manicule.app.ports`, and
:class:`~manicule.app.runtime.Runtime` — the production implementation of them — is imported
by whoever is actually starting a system.
"""

from __future__ import annotations

from manicule.app.bind import Bind, is_every_interface, is_loopback, resolve_bind
from manicule.app.dispatch import error_info, run_op
from manicule.app.ports import (
    Answering,
    Backend,
    DocumentSurface,
    Ingesting,
    Maintenance,
    Retrieving,
)
from manicule.app.results import CONTRACT_VERSION, Envelope, ErrorInfo, Payload, failed, succeeded
from manicule.app.service import DEFAULT_SOURCE, ApplicationService, AskAside, hardware
from manicule.app.tenancy import CrossWorkspaceError, belongs_to, require_owned, require_owns

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_SOURCE",
    "Answering",
    "ApplicationService",
    "AskAside",
    "Backend",
    "Bind",
    "CrossWorkspaceError",
    "DocumentSurface",
    "Envelope",
    "ErrorInfo",
    "Ingesting",
    "Maintenance",
    "Payload",
    "Retrieving",
    "belongs_to",
    "error_info",
    "failed",
    "hardware",
    "is_every_interface",
    "is_loopback",
    "require_owned",
    "require_owns",
    "resolve_bind",
    "run_op",
    "succeeded",
]
