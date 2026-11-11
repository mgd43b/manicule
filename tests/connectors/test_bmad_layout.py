"""BMAD's output folders, read from a path and nothing else.

Every table here is a layout BMAD itself writes (the 6.x line and the 7.x ticket tree), and the
negative cases matter as much as the positive ones: this is enrichment layered over every Git
repository an installation indexes, so a rule that fires on a folder that merely shares a name with
one of BMAD's would label an unrelated project's documents as BMAD artifacts.
"""

from __future__ import annotations

import pytest

from manicule.connectors import bmad_layout
from manicule.connectors.bmad_layout import V6, V7, BmadFacts, classify, describe, detect, v7_roots

V7_TREE = (
    "_bmad-output/initiative-checkout/initiative-checkout.md",
    "_bmad-output/initiative-checkout/prd-checkout/prd-checkout.md",
    "_bmad-output/initiative-checkout/tickets/tickets.toml",
    "_bmad-output/initiative-checkout/tickets/epic-cart-rules/epic-cart-rules.md",
    "_bmad-output/initiative-checkout/tickets/epic-cart-rules/epic-cart-rules-retrospective.md",
    "_bmad-output/initiative-checkout/tickets/epic-cart-rules/ux-cart-rules/ux-cart-rules.md",
    "_bmad-output/initiative-checkout/tickets/epic-cart-rules/tickets/story-cart-service-plan.md",
    "_bmad-output/initiative-checkout/tickets/epic-cart-rules/tickets/story-cart-ui-shell.md",
    "_bmad-output/backlog/bug-checkout-total-ignores-discount-codes.md",
)


def test_a_v7_tree_is_recognized_and_each_file_says_what_it_is() -> None:
    layouts = detect(V7_TREE)
    roots = v7_roots(V7_TREE)

    assert layouts == {V7}
    got = {path: classify(path, layouts=layouts, v7_roots=roots) for path in V7_TREE}
    assert got["_bmad-output/initiative-checkout/initiative-checkout.md"] == BmadFacts(
        V7, "initiative", "checkout"
    )
    assert got["_bmad-output/initiative-checkout/prd-checkout/prd-checkout.md"] == BmadFacts(
        V7, "prd", "checkout"
    )
    epic = "_bmad-output/initiative-checkout/tickets/epic-cart-rules/"
    assert got[f"{epic}epic-cart-rules.md"] == BmadFacts(V7, "epic", "checkout", "cart-rules")
    assert got[f"{epic}epic-cart-rules-retrospective.md"] == BmadFacts(
        V7, "retrospective", "checkout", "cart-rules"
    )
    assert got[f"{epic}ux-cart-rules/ux-cart-rules.md"] == BmadFacts(
        V7, "ux-design", "checkout", "cart-rules"
    )
    assert got[f"{epic}tickets/story-cart-service-plan.md"] == BmadFacts(
        V7, "story-plan", "checkout", "cart-rules"
    )
    assert got[f"{epic}tickets/story-cart-ui-shell.md"] == BmadFacts(
        V7, "story", "checkout", "cart-rules"
    )
    assert got["_bmad-output/backlog/bug-checkout-total-ignores-discount-codes.md"] == BmadFacts(
        V7, "bug"
    )
    assert got["_bmad-output/initiative-checkout/tickets/tickets.toml"] is None


def test_the_output_folder_may_be_the_root_of_the_repository() -> None:
    """A v7 store is often a repository of its own, and then there is no ``_bmad-output`` at all.

    Reading where BMAD was told to write would find nothing here. The shapes inside the folder are
    what is matched, at any depth.
    """
    tree = ("initiative-loyalty/initiative-loyalty.md", "backlog/story-points.md")
    layouts = detect(tree)

    assert layouts == {V7}
    assert classify(tree[0], layouts=layouts, v7_roots=v7_roots(tree)) == BmadFacts(
        V7, "initiative", "loyalty"
    )
    assert classify(tree[1], layouts=layouts, v7_roots=v7_roots(tree)) == BmadFacts(V7, "story")


V6_TREE = (
    "_bmad-output/planning-artifacts/prd.md",
    "_bmad-output/planning-artifacts/architecture.md",
    "_bmad-output/planning-artifacts/epics.md",
    "_bmad-output/planning-artifacts/epics/epic-1.md",
    "_bmad-output/planning-artifacts/ux-design-specification.md",
    "_bmad-output/planning-artifacts/product-brief-checkout-2026.md",
    "_bmad-output/planning-artifacts/project-context.md",
    "_bmad-output/planning-artifacts/notes.md",
    "_bmad-output/implementation-artifacts/1-2-user-auth.md",
    "_bmad-output/implementation-artifacts/stories/story-checkout.md",
    "_bmad-output/implementation-artifacts/epic-1-retro-2026-01-01.md",
    "_bmad-output/implementation-artifacts/sprint-status.yaml",
)


@pytest.mark.parametrize(
    ("path", "kind", "epic"),
    [
        ("_bmad-output/planning-artifacts/prd.md", "prd", None),
        ("_bmad-output/planning-artifacts/architecture.md", "architecture", None),
        ("_bmad-output/planning-artifacts/epics.md", "epic", None),
        ("_bmad-output/planning-artifacts/epics/epic-1.md", "epic", None),
        ("_bmad-output/planning-artifacts/ux-design-specification.md", "ux-design", None),
        ("_bmad-output/planning-artifacts/product-brief-checkout-2026.md", "product-brief", None),
        ("_bmad-output/planning-artifacts/project-context.md", "project-context", None),
        ("_bmad-output/planning-artifacts/notes.md", "planning", None),
        ("_bmad-output/implementation-artifacts/1-2-user-auth.md", "story", "1"),
        ("_bmad-output/implementation-artifacts/stories/story-checkout.md", "story", None),
        ("_bmad-output/implementation-artifacts/2026-01-standup.md", "implementation", None),
        ("_bmad-output/implementation-artifacts/epic-1-retro-2026-01-01.md", "retrospective", None),
    ],
)
def test_a_v6_tree_is_recognized_and_each_file_says_what_it_is(
    path: str, kind: str, epic: str | None
) -> None:
    layouts = detect(V6_TREE)

    assert layouts == {V6}
    assert classify(path, layouts=layouts) == BmadFacts(V6, kind, epic=epic)


def test_sprint_status_alone_is_evidence_of_a_v6_layout() -> None:
    """It is YAML and so never admitted by default, which is exactly why detection is asked of
    the whole tree rather than of what a source admits."""
    assert detect(["output/sprint-status.yaml"]) == {V6}


def test_a_folder_that_merely_shares_a_name_is_not_bmad() -> None:
    """``backlog/`` is a word every project uses, and so are ``prd``, ``epics`` and ``stories``."""
    tree = (
        "docs/backlog/ideas.md",
        "docs/stories/bedtime.md",
        "docs/prd-notes.md",
        "docs/epics.md",
        "README.md",
    )
    layouts = detect(tree)

    assert layouts == frozenset()
    assert describe(layouts) == "none"
    assert all(classify(path, layouts=V6_AND_V7, v7_roots=v7_roots(tree)) is None for path in tree)


V6_AND_V7 = frozenset({V6, V7})


def test_a_backlog_beside_no_initiative_is_not_a_ticket_store() -> None:
    tree = ("elsewhere/initiative-x/initiative-x.md", "docs/backlog/bug-report.md")

    assert classify(tree[1], layouts=detect(tree), v7_roots=v7_roots(tree)) is None


def test_a_migrated_repository_can_hold_both_layouts_and_each_file_is_judged_alone() -> None:
    """``bmad migrate method`` moves v6 files into the v7 tree, so mid-move both layouts exist."""
    tree = (*V6_TREE, *V7_TREE)
    layouts = detect(tree)

    assert layouts == V6_AND_V7
    assert describe(layouts) == "bmad-v6+bmad-v7"
    roots = v7_roots(tree)
    assert classify(V6_TREE[0], layouts=layouts, v7_roots=roots) == BmadFacts(V6, "prd")
    assert classify(V7_TREE[1], layouts=layouts, v7_roots=roots) == BmadFacts(V7, "prd", "checkout")


def test_a_configured_layout_applies_only_its_own_rules() -> None:
    roots = v7_roots(V7_TREE)

    assert classify(V6_TREE[0], layouts=frozenset({V7}), v7_roots=roots) is None
    assert classify(V7_TREE[1], layouts=frozenset({V6}), v7_roots=roots) is None
    assert classify(V7_TREE[1], layouts=frozenset(), v7_roots=roots) is None


def test_only_markdown_is_classified() -> None:
    """The rules are about documents; a TOML manifest in the same folder is not one."""
    layouts = detect(V7_TREE)

    assert classify(V7_TREE[2], layouts=layouts, v7_roots=v7_roots(V7_TREE)) is None


def test_the_layout_names_are_the_ones_configuration_accepts() -> None:
    assert set(bmad_layout.LAYOUTS) == {"auto", "bmad-v6", "bmad-v7", "none"}
