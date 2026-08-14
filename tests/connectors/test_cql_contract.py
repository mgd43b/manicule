"""The documented CQL, built by the real builders and compared to what the documentation says.

``docs/connectors/confluence.md`` shows the query each deployment sends. Prose and code drift
apart the moment they are two records of one fact, and this repository has caught that happening
three times — a documented limit that stopped matching the code, a §4 paragraph naming the wrong
source for a breadcrumb, a §10 row asserting `status` was accepted on a deployment that rejects
it. So the documentation's blocks are **the** copy, and this file executes them.

**Not a second table of query strings.** Nothing here spells a query out. Each expectation is
read from the marked fenced block in the documentation, and the builder is asked to produce it —
so editing one without the other fails, in whichever direction the edit was made.

The blocks are found by an HTML comment marker rather than by position or by nearby prose,
because a section reordered or reworded should not silently stop being checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from manicule.connectors import cql

DOC = Path(__file__).resolve().parents[2] / "docs/connectors/confluence.md"

_BLOCK = re.compile(
    r"<!--\s*cql:(?P<deployment>cloud|server):(?P<name>[a-z-]+)\s*-->\s*```cql\n(?P<query>.*?)\n```",
    re.DOTALL,
)

SPACE = "ENG"
SINCE = "2026/08/09 14:25"
ROOT = "100100"


def documented() -> dict[tuple[str, str], str]:
    """Every marked CQL block in the connector documentation, by deployment and name.

    Raises:
        AssertionError: The documentation carries no marked blocks at all. That is the failure
            worth being loud about — a file rewritten without the markers would otherwise make
            every test below pass by having nothing to check.
    """
    found = {
        (match["deployment"], match["name"]): " ".join(match["query"].split())
        for match in _BLOCK.finditer(DOC.read_text(encoding="utf-8"))
    }
    assert found, f"{DOC} carries no <!-- cql:<deployment>:<name> --> blocks to check against"
    return found


def _documented(deployment: str, name: str) -> str:
    blocks = documented()
    key = (deployment, name)
    if key not in blocks:
        pytest.fail(
            f"{DOC} has no <!-- cql:{deployment}:{name} --> block. Present: "
            f"{sorted(f'{d}:{n}' for d, n in blocks)}"
        )
    return blocks[key]


@pytest.mark.parametrize("deployment", ["cloud", "server"])
def test_the_full_sync_query_is_the_one_the_documentation_shows(deployment: str) -> None:
    built = cql.content_query(
        SPACE, current_only=deployment == "cloud", types=("page", "attachment")
    )

    assert built == _documented(deployment, "full")


@pytest.mark.parametrize("deployment", ["cloud", "server"])
def test_the_incremental_query_is_the_one_the_documentation_shows(deployment: str) -> None:
    built = cql.content_query(
        SPACE, current_only=deployment == "cloud", types=("page", "attachment"), since=SINCE
    )

    assert built == _documented(deployment, "incremental")


@pytest.mark.parametrize("deployment", ["cloud", "server"])
def test_the_subtree_query_is_the_one_the_documentation_shows(deployment: str) -> None:
    built = cql.content_query(
        SPACE,
        current_only=deployment == "cloud",
        subtree=cql.subtree_clause((ROOT,), include_roots=True),
    )

    assert built == _documented(deployment, "subtree")


def test_every_documented_block_is_claimed_by_a_test_above() -> None:
    """A block added to the documentation and to nothing else is a claim nobody checks.

    The failure it prevents is the cheerful one: somebody documents a third deployment's query,
    the file still passes, and the new paragraph is prose with no relationship to the code.
    """
    checked = {
        ("cloud", "full"),
        ("server", "full"),
        ("cloud", "incremental"),
        ("server", "incremental"),
        ("cloud", "subtree"),
        ("server", "subtree"),
    }

    assert set(documented()) == checked


# --- the matrix, deployment by deployment ---------------------------------------------------


CURRENT = "status = current"


def _cases(current_only: bool) -> dict[str, str]:
    """One query per shape this connector actually builds, for one deployment."""
    subtree_with_roots = cql.subtree_clause((ROOT,), include_roots=True)
    subtree_without = cql.subtree_clause((ROOT,), include_roots=False)
    return {
        "pages only": cql.content_query(SPACE, current_only=current_only),
        "pages and attachments": cql.content_query(
            SPACE, current_only=current_only, types=("page", "attachment")
        ),
        "incremental": cql.content_query(SPACE, current_only=current_only, since=SINCE),
        "reconciliation, unordered": cql.content_query(
            SPACE, current_only=current_only, types=("page", "attachment"), ordered=False
        ),
        "attachments only, unordered": cql.content_query(
            SPACE, current_only=current_only, types=("attachment",), ordered=False
        ),
        "subtree including roots": cql.content_query(
            SPACE, current_only=current_only, subtree=subtree_with_roots
        ),
        "subtree excluding roots": cql.content_query(
            SPACE, current_only=current_only, subtree=subtree_without
        ),
        "title lookup": cql.title_query(SPACE, "Retry Policy", current_only=current_only),
    }


def test_every_cloud_query_carries_the_status_clause_exactly_once() -> None:
    """Once: a clause written twice is not harmless, it is a query nobody wrote."""
    for name, query in _cases(current_only=True).items():
        assert query.count(CURRENT) == 1, f"{name}: {query}"
        assert query.count("status") == 1, f"{name}: {query}"


def test_no_server_query_mentions_status_at_all() -> None:
    """The field the standard Data Center content-search resource rejects, in any form."""
    for name, query in _cases(current_only=False).items():
        assert "status" not in query, f"{name}: {query}"


def test_dropping_the_status_clause_changes_nothing_else() -> None:
    """Type, space, subtree, last-modified, quoting and ordering all survive unchanged.

    Asserted by reconstructing one deployment's query from the other's rather than by listing
    the clauses again — a list would be a third copy of the same facts, and the point of this
    file is that there are not supposed to be copies.
    """
    cloud = _cases(current_only=True)
    server = _cases(current_only=False)

    assert set(cloud) == set(server)
    for name, with_status in cloud.items():
        stripped = with_status.replace(f" AND {CURRENT}", "", 1)
        assert stripped == server[name], f"{name}: {stripped!r} != {server[name]!r}"


def test_the_status_clause_is_the_only_thing_the_decision_controls() -> None:
    """The builder's own smallest unit, so a future clause cannot be smuggled in beside it."""
    assert cql.status_clause(current_only=True) == ("status = current",)
    assert cql.status_clause(current_only=False) == ()


def test_the_decision_cannot_be_omitted_by_a_new_call_site() -> None:
    """No default, so forgetting is a `TypeError` where it is written.

    A default would be one deployment's answer applied silently to the other, which is how a
    builder acquires eight callers and seven correct ones. This is the guard that makes the
    eighth impossible rather than merely unlikely.
    """
    with pytest.raises(TypeError, match="current_only"):
        cql.content_query(SPACE)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="current_only"):
        cql.title_query(SPACE, "Retry Policy")  # type: ignore[call-arg]
