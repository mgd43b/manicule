"""What BMAD's output folders say about a document, read from its path alone.

A repository that holds BMAD output is full of Markdown whose role is stated in its folder and
filename and nowhere in its text: this is the product requirements document, that is a story, this
is the retrospective of an epic in the checkout initiative. A citation that says
``prd-checkout.md`` is accurate and useless; one that says it is the PRD of the checkout
initiative is a thing a person can act on. This module turns a path into those facts and does
nothing else — it reads no file, runs no command and consults no configuration, so the same tree
gives the same answer on every machine.

**Paths, not configuration.** Both versions let a project move the output folder (``output_folder``
in BMAD's own configuration), and a v7 store is often a repository of its own with the output
folder *as* the root. Reading where BMAD was told to write would mean finding that file, parsing
it and trusting it, and would still say nothing for a repository that holds only the output. The
shapes below are what BMAD writes *inside* whatever folder that is, and they are matched at any
depth, so the answer does not depend on where the folder sits.

**Two layouts, matched by what each writes.**

``bmad-v6`` — the 6.x line (stable as 6.12). Planning documents under ``planning-artifacts/``
    (``prd.md``, ``architecture.md``, ``epics.md`` or ``epics/epic-*.md``, briefs, UX designs) and
    delivery documents under ``implementation-artifacts/`` (stories named ``<epic>-<story>-<slug>``,
    retrospectives, ``sprint-status.yaml``).

``bmad-v7`` — the ticket tree of the 7.x preview. One ``initiative-<slug>/`` folder per body of
    work holding ``initiative-<slug>.md``, ``prd-<slug>/``, ``ux-<slug>/``, ``architecture-<slug>/``
    and ``tickets/``, whose ``epic-<slug>/`` folders hold the epic, its plans
    (``story-<slug>-plan.md``) and its retrospective, plus a ``backlog/`` beside the initiatives
    for standalone tickets.

**These are heuristics over a format that is still moving.** The 7.x line is a preview alpha. A
path that matches neither layout is not a failure: it simply has no BMAD facts, and is indexed
exactly as any other document. A rule that stops matching after BMAD renames a folder costs the
extra citation detail and nothing else, which is why this is enrichment and not a gate on what is
admitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "AUTO",
    "LAYOUTS",
    "NONE",
    "V6",
    "V7",
    "BmadFacts",
    "classify",
    "describe",
    "detect",
]

V6: Final = "bmad-v6"
V7: Final = "bmad-v7"
AUTO: Final = "auto"
NONE: Final = "none"

LAYOUTS: Final = (AUTO, V6, V7, NONE)
"""What ``layout`` may be set to."""

_PLANNING: Final = "planning-artifacts"
_IMPLEMENTATION: Final = "implementation-artifacts"
_INITIATIVE: Final = "initiative-"
_EPIC: Final = "epic-"
_STORY_FILE: Final = re.compile(r"^(?P<epic>\d{1,3})-\d{1,3}(?:-[^/]+)?\.md$")
"""``<epic>-<story>-<slug>``. Bounded to three digits so a dated note (``2026-01-standup.md``)
in the same folder is not read as the story of epic 2026."""

_PLANNING_KINDS: Final = (
    ("prd", "prd"),
    ("architecture", "architecture"),
    ("epic", "epic"),
    ("ux", "ux-design"),
    ("product-brief", "product-brief"),
    ("brief", "product-brief"),
    ("project-context", "project-context"),
    ("brainstorm", "brainstorming"),
    ("research", "research"),
    ("tech-spec", "tech-spec"),
    ("implementation-readiness", "readiness-report"),
)
"""Filename or folder prefixes under ``planning-artifacts/``, most specific first."""

_V7_FOLDER_KINDS: Final = (
    ("prd-", "prd"),
    ("ux-", "ux-design"),
    ("architecture-", "architecture"),
)


@dataclass(frozen=True, slots=True)
class BmadFacts:
    """What a path says about the BMAD document it names."""

    layout: str
    kind: str
    initiative: str | None = None
    epic: str | None = None


def detect(paths: Iterable[str]) -> frozenset[str]:
    """Which layouts the tree shows evidence of, from every path in it.

    Asked of the whole tree and not only of what a source admits: ``tickets.toml`` and
    ``sprint-status.yaml`` are the clearest evidence there is, and they are not Markdown.
    """
    found: set[str] = set()
    for path in paths:
        segments = path.split("/")
        if (
            _PLANNING in segments[:-1]
            or _IMPLEMENTATION in segments[:-1]
            or segments[-1] == "sprint-status.yaml"
        ):
            found.add(V6)
        if _v7_root(segments) is not None or segments[-2:] == ["tickets", "tickets.toml"]:
            found.add(V7)
    return frozenset(found)


def describe(detected: frozenset[str]) -> str:
    """The layouts found as one word, for a log line and a status."""
    if not detected:
        return NONE
    return "+".join(sorted(detected))


def classify(
    path: str, *, layouts: frozenset[str], v7_roots: frozenset[str] = frozenset()
) -> BmadFacts | None:
    """What ``path`` is in BMAD terms, or ``None`` when it matches no rule of an enabled layout.

    ``v7_roots`` are the folders that directly hold an ``initiative-*`` folder, which is what lets
    a ``backlog/`` beside them be recognized without every folder in the world called ``backlog``
    being mistaken for one.
    """
    segments = path.split("/")
    if V7 in layouts:
        facts = _v7(segments, v7_roots)
        if facts is not None:
            return facts
    if V6 in layouts:
        return _v6(segments)
    return None


def v7_roots(paths: Iterable[str]) -> frozenset[str]:
    """The folders directly holding an initiative, as ``/``-joined prefixes (``""`` is the root)."""
    roots: set[str] = set()
    for path in paths:
        root = _v7_root(path.split("/"))
        if root is not None:
            roots.add(root)
    return frozenset(roots)


def _v7_root(segments: list[str]) -> str | None:
    """The folder holding ``initiative-<x>/initiative-<x>.md``, if this path is that file."""
    if len(segments) < 2:  # noqa: PLR2004 - a folder and a file
        return None
    folder, name = segments[-2], segments[-1]
    if folder.startswith(_INITIATIVE) and len(folder) > len(_INITIATIVE) and name == f"{folder}.md":
        return "/".join(segments[:-2])
    return None


def _slug(segment: str, prefix: str) -> str | None:
    if segment.startswith(prefix) and len(segment) > len(prefix):
        return segment[len(prefix) :]
    return None


def _nearest(folders: list[str], prefix: str) -> str | None:
    for segment in reversed(folders):
        slug = _slug(segment, prefix)
        if slug is not None:
            return slug
    return None


def _v7(segments: list[str], roots: frozenset[str]) -> BmadFacts | None:
    folders, name = segments[:-1], segments[-1]
    if not name.endswith(".md"):
        return None
    stem = name.removesuffix(".md")
    initiative = _nearest(folders, _INITIATIVE)
    if initiative is None:
        # A standalone ticket: a file in a ``backlog/`` that sits beside initiatives.
        beside_initiatives = (
            bool(folders) and folders[-1] == "backlog" and "/".join(folders[:-1]) in roots
        )
        return BmadFacts(V7, _ticket_kind(stem, default="backlog")) if beside_initiatives else None
    epic = _nearest(folders, _EPIC)
    return BmadFacts(V7, _v7_kind(folders, name, initiative, epic), initiative, epic)


def _v7_kind(folders: list[str], name: str, initiative: str, epic: str | None) -> str:
    """What a file inside an initiative is, from its own name and the folders around it."""
    return (
        _v7_named(folders[-1], name, initiative, epic)
        or _v7_placed(folders, name.removesuffix(".md"))
        or "initiative-document"
    )


def _v7_named(here: str, name: str, initiative: str, epic: str | None) -> str | None:
    """A document that is the one its own folder is named for, or an epic's retrospective."""
    stem = name.removesuffix(".md")
    if name == ".memlog.md":
        return "memlog"
    if stem == here == f"{_INITIATIVE}{initiative}":
        return "initiative"
    if epic is not None and stem == f"{_EPIC}{epic}-retrospective":
        return "retrospective"
    if epic is not None and stem == here == f"{_EPIC}{epic}":
        return "epic"
    return None


def _v7_placed(folders: list[str], stem: str) -> str | None:
    """A document known by the folder it sits in (``prd-*``) or by its own prefix (``story-*``)."""
    for segment in reversed(folders):
        for prefix, kind in _V7_FOLDER_KINDS:
            if _slug(segment, prefix) is not None:
                return kind
    if stem.startswith("story-") and stem.endswith("-plan"):
        return "story-plan"
    if stem.startswith(("story-", "bug-", "spike-")):
        return _ticket_kind(stem, default="story")
    return None


def _ticket_kind(stem: str, *, default: str) -> str:
    for prefix in ("story", "bug", "spike", "epic"):
        if stem.startswith(f"{prefix}-"):
            return prefix
    return default


_V6_DELIVERY_KINDS: Final = (
    ("retro", "retrospective"),
    ("tech-spec", "tech-spec"),
    ("review", "review"),
)


def _v6(segments: list[str]) -> BmadFacts | None:
    folders, name = segments[:-1], segments[-1]
    if not name.endswith(".md"):
        return None
    for index, segment in enumerate(folders):
        below = folders[index + 1 :]
        if segment == _PLANNING:
            return BmadFacts(V6, _planning_kind((below[0] if below else name).lower()))
        if segment == _IMPLEMENTATION:
            return _v6_delivery(name, below)
    return None


def _v6_delivery(name: str, below: list[str]) -> BmadFacts:
    """A file under ``implementation-artifacts/``: a story, or what was written about one."""
    matched = _STORY_FILE.match(name)
    if matched is not None:
        return BmadFacts(V6, "story", epic=matched.group("epic"))
    if below and below[0].lower() == "stories":
        return BmadFacts(V6, "story")
    lowered = name.lower()
    kind = next(
        (kind for marker, kind in _V6_DELIVERY_KINDS if marker in lowered), "implementation"
    )
    return BmadFacts(V6, kind)


def _planning_kind(label: str) -> str:
    for prefix, kind in _PLANNING_KINDS:
        if label.startswith(prefix) or (prefix == "brief" and "brief" in label):
            return kind
    return "planning"
