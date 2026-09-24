"""An administrator's search spanning workspaces: the policy around the merge, in the service.

The merge itself is retrieval's and ``tests/retrieval/test_spanning.py`` holds it to the
settled rule. What is held here is everything the service decides before and after it — who may
ask, how many workspaces, which collections, and the identity of every hit in the workspace it
came from — against the same deliberately broken stores ``test_tenancy.py`` uses, because a
guard run only against correct stores passes whether or not it exists.

``tests/app/test_cross_workspace_runtime.py`` does the same end to end, against a real data
directory holding several workspaces.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from manicule.app.caller import Caller, acting_as
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.app.tenancy import CrossWorkspaceError
from manicule.config.settings import Role, Settings
from manicule.core.errors import ConfigError, PolicyError, UnknownEntityError
from manicule.core.retrieval import Candidate
from tests.app.fakes import (
    FakeBackend,
    FakeOrganization,
    FakeRetriever,
    FakeStore,
    LeakyStore,
    SpanningBackend,
    SpanningRetriever,
    make_chunk,
    make_document,
    spanning_backend,
)

ALPHA = "alpha"
BETA = "beta"
GAMMA = "gamma"
FOREIGN_TITLE = "Gamma salary bands"
FOREIGN_TEXT = "Everyone in gamma earns exactly nine hundred thousand"
"""Content no workspace in these searches may see. Distinctive, so a leak is greppable."""


def _backend(**settings: Any) -> SpanningBackend:
    """Alpha serving, beta beside it, each holding one runbook; the retriever spans both."""
    return spanning_backend(ALPHA, BETA, **settings)


def _spanning(backend: SpanningBackend) -> SpanningRetriever:
    return backend.spanning


# --- when a search is an ordinary one ----------------------------------------------------------


@pytest.mark.parametrize("workspaces", [None, [], [ALPHA], [f" {ALPHA} ", ALPHA]])
async def test_naming_no_workspace_or_only_this_one_is_the_ordinary_search(
    workspaces: list[str] | None,
) -> None:
    """The ordinary path, byte for byte, for every spelling of "just this workspace".

    A viewer asks: if any of these reached the spanning path, the admin check would refuse a
    search the viewer is entitled to, and the surfaces' floor — which asks the same question
    through :meth:`crosses_workspaces` — would refuse it first.
    """
    backend = _backend()
    service = ApplicationService(backend)

    with acting_as(Caller(role=Role.VIEWER)):
        assert not service.crosses_workspaces(workspaces)
        result = await service.search("runbook", workspaces=workspaces)

    assert _spanning(backend).seen_across == []
    assert [query.filter.workspace_ids for query in backend.retriever_.seen] == [frozenset({ALPHA})]
    assert result.workspaces == (ALPHA,)
    assert [hit.workspace for hit in result.hits] == [ALPHA]


async def test_a_blank_workspace_name_is_refused_rather_than_dropped() -> None:
    """Dropping it would search fewer workspaces than the caller believes they named."""
    service = ApplicationService(_backend())

    with pytest.raises(ConfigError, match="blank"):
        await service.search("runbook", workspaces=[ALPHA, "  "])


# --- who may, and how many ---------------------------------------------------------------------


@pytest.mark.parametrize("role", [Role.VIEWER, Role.MEMBER])
async def test_a_caller_short_of_admin_cannot_search_across_workspaces(role: Role) -> None:
    """Refused before anything is opened, so a refusal reveals nothing about the others.

    Checked in the service, which is where it has to be for the rule to hold on a surface that
    forgets to ask — the network surfaces' own floor is the courtesy that turns it into their
    ordinary 403 before the service is reached.
    """
    backend = _backend()
    service = ApplicationService(backend)

    with acting_as(Caller(role=role)), pytest.raises(PolicyError, match="'admin' role"):
        await service.search("runbook", workspaces=[ALPHA, BETA])

    assert backend.opened == []
    assert _spanning(backend).seen_across == []


@pytest.mark.parametrize("caller", [Caller(role=Role.ADMIN), Caller()])
async def test_an_administrator_and_the_local_operator_can(caller: Caller) -> None:
    """The positive control: the admin role, and the operator at this machine who holds all."""
    service = ApplicationService(_backend())

    with acting_as(caller):
        result = await service.search("runbook", workspaces=[ALPHA, BETA])

    assert result.workspaces == (ALPHA, BETA)


def _person(backend: SpanningBackend, *, member_of: dict[str, bool]) -> str:
    """One signed-in person, an admin of alpha, with the other memberships named.

    ``member_of`` maps a workspace to whether that membership is disabled.
    """
    users = backend.users_
    users.people["u-ada"] = {"provider": "github", "subject": "1", "email": None, "name": "Ada"}
    users.memberships[(ALPHA, "u-ada")] = {"role": "admin", "disabled": False}
    for workspace, disabled in member_of.items():
        users.memberships[(workspace, "u-ada")] = {"role": "viewer", "disabled": disabled}
    return "u-ada"


async def test_administering_this_workspace_is_not_standing_in_another() -> None:
    """A person who administers alpha and was never admitted to beta cannot read beta.

    The admin role is a relationship with the serving workspace. Without this check, anybody an
    administrator of one workspace could become would read every corpus on the installation.
    Refused before anything is opened, so the refusal reveals nothing about beta either.
    """
    backend = _backend()
    service = ApplicationService(backend)
    person = _person(backend, member_of={})

    with (
        acting_as(Caller(role=Role.ADMIN, user_id=person)),
        pytest.raises(PolicyError, match="no enabled membership of 'beta'"),
    ):
        await service.search("runbook", workspaces=[ALPHA, BETA])

    assert backend.opened == []
    assert _spanning(backend).seen_across == []


async def test_a_disabled_membership_is_no_standing_either() -> None:
    """Disabling somebody in beta must stop them reading beta by way of alpha."""
    backend = _backend()
    service = ApplicationService(backend)
    person = _person(backend, member_of={BETA: True})

    with (
        acting_as(Caller(role=Role.ADMIN, user_id=person)),
        pytest.raises(PolicyError, match="'beta'"),
    ):
        await service.search("runbook", workspaces=[ALPHA, BETA])


async def test_a_person_admitted_to_every_named_workspace_may_span_them() -> None:
    """The positive control: any enabled role there is standing, since roles gate actions."""
    backend = _backend()
    service = ApplicationService(backend)
    person = _person(backend, member_of={BETA: False})

    with acting_as(Caller(role=Role.ADMIN, user_id=person)):
        result = await service.search("runbook", workspaces=[ALPHA, BETA])

    assert result.workspaces == (ALPHA, BETA)


async def test_a_key_with_no_owner_is_the_operators_delegate() -> None:
    """Only the operator at this machine mints an ownerless key, so it spans as they would."""
    backend = _backend()
    service = ApplicationService(backend)

    with acting_as(Caller(role=Role.ADMIN, key_id="k-operator")):
        result = await service.search("runbook", workspaces=[ALPHA, BETA])

    assert result.workspaces == (ALPHA, BETA)


@pytest.mark.parametrize(("limit", "named"), [(3, 3), (8, 8)])
async def test_the_configured_bound_admits_exactly_its_own_count(limit: int, named: int) -> None:
    """The edge on the accepting side: ``cross_workspace_limit`` workspaces is allowed."""
    backend = _backend(rag={"cross_workspace_limit": limit})
    names = [ALPHA, BETA] + [f"extra-{index}" for index in range(named - 2)]
    for name in names[2:]:
        backend.others[name] = (FakeStore(workspace_id=name), FakeOrganization(workspace_id=name))

    result = await ApplicationService(backend).search("runbook", workspaces=names)

    assert result.workspaces == tuple(names)


@pytest.mark.parametrize("limit", [3, 8])
async def test_one_workspace_past_the_bound_is_refused_before_any_is_opened(limit: int) -> None:
    """The edge on the refusing side, and before the work it exists to bound has started."""
    backend = _backend(rag={"cross_workspace_limit": limit})
    names = [ALPHA, BETA] + [f"extra-{index}" for index in range(limit - 1)]

    with pytest.raises(PolicyError, match=f"allows {limit}"):
        await ApplicationService(backend).search("runbook", workspaces=names)

    assert backend.opened == []


@pytest.mark.parametrize(("value", "valid"), [(1, False), (2, True), (64, True), (65, False)])
def test_the_bound_itself_is_bounded(value: int, valid: bool) -> None:
    """Below two there is nothing to span; above 64 one request is a sweep of the installation."""
    if valid:
        assert Settings(rag={"cross_workspace_limit": value}).rag.cross_workspace_limit == value  # pyright: ignore[reportArgumentType]
    else:
        with pytest.raises(ValueError, match="cross_workspace_limit"):
            Settings(rag={"cross_workspace_limit": value})  # pyright: ignore[reportArgumentType]


async def test_this_workspace_is_searched_only_when_it_is_named() -> None:
    """``b`` from a process serving ``a`` searches ``b``: a scope never widens by itself."""
    backend = _backend()
    _spanning(backend).across = _spanning(backend).across[:1]

    result = await ApplicationService(backend).search("runbook", workspaces=[BETA])

    query, legs = _spanning(backend).seen_across[0]
    assert legs == [BETA]
    assert query.filter.workspace_ids == frozenset({BETA})
    assert result.workspaces == (BETA,)
    assert [hit.workspace for hit in result.hits] == [BETA]


# --- what a spanning search returns ------------------------------------------------------------


async def test_every_hit_names_the_workspace_it_came_from() -> None:
    """The ranking is the merge's; each line of it carries its own workspace."""
    backend = _backend()

    result = await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])

    assert [(hit.title, hit.workspace) for hit in result.hits] == [
        ("Beta runbook", BETA),
        ("Alpha runbook", ALPHA),
    ]
    assert result.expansions == ()
    assert result.expanded_query == ""


async def test_a_collection_missing_from_one_workspace_refuses_the_whole_search() -> None:
    """A scope that fails is a refusal, and never a narrowing that searches one workspace whole."""
    backend = _backend()
    alpha_runbooks = await backend.organization_.create_collection("runbooks")

    with pytest.raises(UnknownEntityError, match="'runbooks' in workspace 'beta'"):
        await ApplicationService(backend).search(
            "runbook", workspaces=[ALPHA, BETA], collections=["runbooks"]
        )
    assert _spanning(backend).seen_across == []

    beta_runbooks = await backend.others[BETA][1].create_collection("runbooks")
    await ApplicationService(backend).search(
        "runbook", workspaces=[ALPHA, BETA], collections=["runbooks"]
    )
    query, _ = _spanning(backend).seen_across[0]
    assert query.filter.collection_ids == frozenset({alpha_runbooks.id, beta_runbooks.id})


async def test_a_spanning_search_is_logged_where_it_ran_and_audited() -> None:
    """One query-log row, in this workspace, and one audit entry naming what was spanned.

    The audit entry carries the workspaces and the count, never the query: the audit trail
    outlives the workspaces it describes, and query text is user content.
    """
    backend = _backend(security={"audit": {"enabled": True}})

    await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])

    assert [row["query"] for row in backend.telemetry_.queries] == ["runbook"]
    assert [(entry["event_type"], entry["details"]) for entry in backend.telemetry_.audits] == [
        ("search.cross_workspace", {"workspaces": [ALPHA, BETA], "hits": 2})
    ]


async def test_a_backend_that_cannot_open_other_workspaces_refuses_rather_than_narrows() -> None:
    """Searching the one workspace it can open would answer a question nobody asked."""
    backend = FakeBackend(settings=Settings(workspace=ALPHA))

    with pytest.raises(ConfigError, match="cannot open workspaces"):
        await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])


async def test_a_retriever_that_cannot_span_refuses_rather_than_searching_one() -> None:
    """The same refusal from the retrieval side: no fallback to the ordinary search."""
    backend = _backend()
    backend.retriever_ = FakeRetriever()

    with pytest.raises(ConfigError, match="only the workspace it serves"):
        await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])


@pytest.mark.parametrize("operation", ["ask", "ask_stream", "research"])
def test_answering_and_research_cannot_be_asked_to_span_workspaces(operation: str) -> None:
    """Cross-workspace is a search, and only a search.

    An answer sends passages to a model, and a research report runs several retrievals into
    one; spanning tenants in either is a disclosure decision nobody has made. They take no
    ``workspaces`` argument, and the retrieval they reach refuses a query naming more than one
    (``tests/retrieval/test_spanning.py``).
    """
    assert "workspaces" not in inspect.signature(getattr(ApplicationService, operation)).parameters


# --- identity, per workspace -------------------------------------------------------------------


def _leaking_beta() -> tuple[SpanningBackend, Candidate]:
    """Beta's store ignores its scope and holds a gamma document; beta's search returns it."""
    backend = _backend()
    leaky = LeakyStore(workspace_id=BETA)
    leaky.add(make_document(BETA, source_id="beta.md", title="Beta runbook"))
    foreign = make_document(GAMMA, source="hr", source_id="bands.md", title=FOREIGN_TITLE)
    leaky.add(foreign, make_chunk(foreign, text=FOREIGN_TEXT))
    backend.others[BETA] = (leaky, backend.others[BETA][1])
    leaked = Candidate(chunk=make_chunk(foreign, text=FOREIGN_TEXT), score=0.95)
    _spanning(backend).across = [((BETA,), leaked), *_spanning(backend).across]
    return backend, leaked


async def test_a_leg_whose_store_ignored_its_scope_is_refused_whole() -> None:
    """Beta's store hands back gamma's row; beta's identity check is what refuses it.

    A digest of ``(workspace, source, source_id)`` cannot be satisfied by another workspace's
    document, so the check fires however the store behaves.
    """
    backend, _ = _leaking_beta()

    with pytest.raises(CrossWorkspaceError, match="'beta'"):
        await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])


async def test_nothing_of_the_leaked_workspace_reaches_the_serialized_result() -> None:
    """The refusal itself must not carry what it refused."""
    backend, leaked = _leaking_beta()
    service = ApplicationService(backend)

    envelope = await run_op(
        "search", service.workspace, lambda: service.search("runbook", workspaces=[ALPHA, BETA])
    )

    assert not envelope.ok
    serialized = json.dumps(envelope.as_json())
    for fragment in (FOREIGN_TITLE, FOREIGN_TEXT, leaked.chunk.document_id, GAMMA):
        assert fragment not in serialized


async def test_a_hit_attributed_to_the_wrong_workspace_is_refused() -> None:
    """Correct stores, wrong attribution: alpha cannot vouch for beta's document."""
    backend = _backend()
    retriever = _spanning(backend)
    retriever.across = [((ALPHA,), candidate) for _, candidate in retriever.across]

    with pytest.raises(CrossWorkspaceError, match="'alpha' cannot see"):
        await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])


@pytest.mark.parametrize("claimed", [(), (ALPHA, BETA), (GAMMA,)])
async def test_a_hit_claimed_by_no_workspace_or_by_two_is_refused(claimed: tuple[str, ...]) -> None:
    """A passage belongs to exactly one workspace this search opened, or something leaked."""
    backend = _backend()
    retriever = _spanning(backend)
    retriever.across = [(claimed, retriever.across[0][1]), *retriever.across[1:]]

    with pytest.raises(CrossWorkspaceError, match="nothing was returned"):
        await ApplicationService(backend).search("runbook", workspaces=[ALPHA, BETA])
