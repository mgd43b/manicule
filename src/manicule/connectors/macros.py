"""Macros that pull content from other pages, and what it takes to see it.

``include`` and ``excerpt-include`` render another page's content inline. A reader sees that
content on the page; a body fetched from the API carries only the macro node. Unresolved, the
text is **missing from the index while appearing present in the UI** — a gap that is invisible
from both ends, which is why ``docs/connectors/confluence.md`` §5 calls macros the place where
content hides.

Resolution is bounded in two ways, and both are required rather than defensive:

- **A depth limit.** A page including a page that includes a page is ordinary. Deeper is a
  template that has grown a loop somebody has not noticed.
- **Cycle detection.** Two pages that include each other are not exotic — an overview page and
  a detail page cross-including is a normal authoring mistake — and without a check the
  expansion never terminates.

**Nothing unresolved is dropped in silence.** Every macro that could not be expanded is
recorded on the document with the reason, because a chunk that is quietly short is the failure
this module exists to prevent, and replacing it with quietly *invented* text would be worse:
manicule's own words would end up inside a quotation attributed to the source.

Both body formats are handled here, because both deployments have the problem. Atlassian
Document Format is a node tree and is edited as one. Storage format is XHTML, and it is read
with :mod:`html.parser` — a real parser, never a regular expression — which reports the exact
span of each macro element so the raw source can be spliced without being re-serialised. A
round trip through any serialiser would rewrite markup that was not part of the edit.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import cast, override

from pydantic import JsonValue

__all__ = [
    "INCLUDE_MACROS",
    "IncludedPage",
    "Lookup",
    "MacroReport",
    "MacroTarget",
    "Unresolved",
    "excerpt_of_adf",
    "excerpt_of_storage",
    "find_adf_macros",
    "find_storage_macros",
    "resolve_adf",
    "resolve_storage",
    "storage_macros",
    "unresolved_because",
]

INCLUDE_MACROS = frozenset({"include", "excerpt-include"})
"""The macros that pull in another page. ``excerpt`` marks content; it does not import any."""

EXCERPT_MACRO = "excerpt"

_ADF_EXTENSIONS = frozenset({"extension", "inlineExtension", "bodiedExtension"})

_TITLE_PARAMETERS = ("", "page", "pageTitle", "name")
"""Where an include macro's target title has been seen to live, most common first.

The unnamed parameter is the macro's default one and carries the target in the editor's own
output; the others appear in bodies written by templates and by older editors. Reading only
the first would leave those pages short by exactly the content the macro was there to add.
"""

_ID_PARAMETERS = ("contentId", "pageId")


@dataclass(frozen=True, slots=True)
class MacroTarget:
    """What an include macro points at."""

    macro: str
    title: str = ""
    space: str = ""
    content_id: str = ""

    @property
    def wants_excerpt(self) -> bool:
        return self.macro == "excerpt-include"

    def describe(self) -> str:
        if self.content_id:
            return f"content id {self.content_id}"
        where = f" in space {self.space}" if self.space else ""
        return f"{self.title!r}{where}" if self.title else "an unnamed page"


@dataclass(frozen=True, slots=True)
class IncludedPage:
    """A page an include macro resolved to, with whichever body format was fetched."""

    page_id: str
    title: str
    adf: Mapping[str, object] | None = None
    storage: str | None = None


@dataclass(frozen=True, slots=True)
class Unresolved:
    """A macro that was left unexpanded, and why."""

    macro: str
    target: str
    reason: str

    def as_metadata(self) -> dict[str, JsonValue]:
        return {"macro": self.macro, "target": self.target, "reason": self.reason}


@dataclass(slots=True)
class MacroReport:
    """What macro expansion did, for the document's metadata.

    Collected rather than logged: "this page renders content from three others" is provenance,
    and "one of them could not be read" is the thing somebody has to be able to find.
    """

    included: list[str] = field(default_factory=list[str])
    """Page ids whose content was spliced in, in the order they were resolved."""

    unresolved: list[Unresolved] = field(default_factory=list[Unresolved])

    def as_metadata(self) -> dict[str, JsonValue]:
        included: list[JsonValue] = list(self.included)
        unresolved: list[JsonValue] = [item.as_metadata() for item in self.unresolved]
        return {"included_pages": included, "unresolved_macros": unresolved}


type Lookup = Callable[[MacroTarget], Coroutine[object, object, IncludedPage | None]]
"""How the resolver finds a page. Supplied by the connector, because finding one is a search."""


# --- Atlassian Document Format ------------------------------------------------------------


async def resolve_adf(
    document: Mapping[str, object],
    *,
    lookup: Lookup,
    depth_limit: int,
    report: MacroReport,
    path: Sequence[str] = (),
    depth: int = 0,
) -> dict[str, object]:
    """``document`` with every include macro replaced by the content it names.

    The included nodes are spliced in at the macro's position, which is where Confluence
    renders them. That matters for citations as well as for text: a heading arriving through
    an include is a heading on the *rendering* page, and Confluence derives its anchor there —
    so the deep link the parser builds from it addresses the page a reader would open.

    Args:
        document: The ADF root node.
        lookup: How to find the page a macro names.
        depth_limit: How far expansion may nest.
        report: Collects what was included and what was not.
        path: Page ids already being expanded, **including the page this body belongs to**.
            A page that includes itself is caught on its first macro rather than after one
            round of duplication.
        depth: How many expansions deep this call already is.
    """
    content = await _resolve_nodes(
        _content_of(document),
        lookup=lookup,
        depth_limit=depth_limit,
        report=report,
        path=path,
        depth=depth,
    )
    resolved = dict(document)
    resolved["content"] = content
    return resolved


async def _resolve_nodes(
    nodes: Sequence[Mapping[str, object]],
    *,
    lookup: Lookup,
    depth_limit: int,
    report: MacroReport,
    path: Sequence[str],
    depth: int,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for node in nodes:
        target = adf_macro_target(node)
        if target is not None:
            out.extend(
                await _expand_adf(
                    target,
                    lookup=lookup,
                    depth_limit=depth_limit,
                    report=report,
                    path=path,
                    depth=depth,
                )
            )
            continue
        children = _content_of(node)
        if not children:
            out.append(node)
            continue
        replaced = dict(node)
        replaced["content"] = await _resolve_nodes(
            children,
            lookup=lookup,
            depth_limit=depth_limit,
            report=report,
            path=path,
            depth=depth,
        )
        out.append(replaced)
    return out


async def _expand_adf(
    target: MacroTarget,
    *,
    lookup: Lookup,
    depth_limit: int,
    report: MacroReport,
    path: Sequence[str],
    depth: int,
) -> list[Mapping[str, object]]:
    """The nodes an include macro contributes: its target's content, or nothing and a reason."""
    if depth >= depth_limit:
        report.unresolved.append(
            Unresolved(target.macro, target.describe(), _too_deep(depth_limit, path))
        )
        return []

    page = await lookup(target)
    if page is None or page.adf is None:
        report.unresolved.append(
            Unresolved(target.macro, target.describe(), _not_found(target, page))
        )
        return []
    if page.page_id in path:
        report.unresolved.append(Unresolved(target.macro, target.describe(), _cycle(page, path)))
        return []

    body = excerpt_of_adf(page.adf) if target.wants_excerpt else _content_of(page.adf)
    if body is None:
        report.unresolved.append(Unresolved(target.macro, target.describe(), _no_excerpt(page)))
        return []

    report.included.append(page.page_id)
    return await _resolve_nodes(
        body,
        lookup=lookup,
        depth_limit=depth_limit,
        report=report,
        path=(*path, page.page_id),
        depth=depth + 1,
    )


def adf_macro_target(node: Mapping[str, object]) -> MacroTarget | None:
    """The include this node is, or ``None`` if it is not one."""
    if _text(node.get("type")) not in _ADF_EXTENSIONS:
        return None
    attrs = _mapping(node.get("attrs"))
    key = _text(attrs.get("extensionKey"))
    if key not in INCLUDE_MACROS:
        return None
    params = _macro_params(attrs)
    return MacroTarget(
        macro=key,
        title=_first(params, _TITLE_PARAMETERS),
        space=_text(params.get("spaceKey")),
        content_id=_first(params, _ID_PARAMETERS),
    )


def excerpt_of_adf(document: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    """The content inside a page's ``excerpt`` macro, or ``None`` when it defines none."""
    for node in _walk_adf(_content_of(document)):
        attrs = _mapping(node.get("attrs"))
        if _text(attrs.get("extensionKey")) != EXCERPT_MACRO:
            continue
        if _text(node.get("type")) not in _ADF_EXTENSIONS:
            continue
        return list(_content_of(node))
    return None


def _walk_adf(nodes: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    found: list[Mapping[str, object]] = []
    for node in nodes:
        found.append(node)
        found.extend(_walk_adf(_content_of(node)))
    return found


def _macro_params(attrs: Mapping[str, object]) -> dict[str, str]:
    """``parameters.macroParams`` flattened to name -> value.

    Each parameter arrives as ``{"value": ...}``. A bare string is accepted too, because
    bodies produced by older editors have been seen with one, and refusing it would drop the
    include rather than the wrapper.
    """
    parameters = _mapping(attrs.get("parameters"))
    macro_params = _mapping(parameters.get("macroParams"))
    flattened: dict[str, str] = {}
    for name, raw in macro_params.items():
        if isinstance(raw, str):
            flattened[name] = raw.strip()
            continue
        value = _mapping(raw).get("value")
        if isinstance(value, str | int):
            flattened[name] = str(value).strip()
    return flattened


def _first(params: Mapping[str, str], names: Sequence[str]) -> str:
    return next((params[name] for name in names if params.get(name)), "")


def _content_of(node: Mapping[str, object]) -> list[Mapping[str, object]]:
    content = node.get("content")
    if not isinstance(content, list):
        return []
    entries = cast("Sequence[object]", content)
    return [cast("Mapping[str, object]", e) for e in entries if isinstance(e, dict)]


def _mapping(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


# --- storage format ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StorageMacro:
    """One ``ac:structured-macro`` element, located exactly within its source."""

    name: str
    start: int
    end: int
    params: Mapping[str, str] = field(default_factory=dict[str, str])
    body: tuple[int, int] | None = None
    """Span of the macro's ``ac:rich-text-body``, when it has one."""

    @property
    def target(self) -> MacroTarget:
        return MacroTarget(
            macro=self.name,
            title=_first(self.params, _TITLE_PARAMETERS),
            space=self.params.get("spacekey", ""),
            content_id=_first(self.params, ("contentid", "pageid")),
        )


async def resolve_storage(
    body: str,
    *,
    lookup: Lookup,
    depth_limit: int,
    report: MacroReport,
    path: Sequence[str] = (),
    depth: int = 0,
) -> str:
    """``body`` with every include macro replaced by the storage-format content it names.

    The replacement is a splice into the original string at spans the parser reported, so
    everything that is not a macro arrives at the parser chain byte-identical to what
    Confluence returned. Reconstructing the document from a parse tree instead would rewrite
    entities, attribute quoting and self-closing tags across the whole page in order to change
    one element.
    """
    macros = [macro for macro in storage_macros(body) if macro.name in INCLUDE_MACROS]
    if not macros:
        return body

    replacements: list[tuple[int, int, str]] = []
    for macro in macros:
        replacements.append(
            (
                macro.start,
                macro.end,
                await _expand_storage(
                    macro.target,
                    lookup=lookup,
                    depth_limit=depth_limit,
                    report=report,
                    path=path,
                    depth=depth,
                ),
            )
        )

    # Applied from the end backwards so that every span still addresses the text it was
    # measured against; splicing forwards would shift each later span by the length change.
    out = body
    for start, end, replacement in sorted(replacements, reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


async def _expand_storage(
    target: MacroTarget,
    *,
    lookup: Lookup,
    depth_limit: int,
    report: MacroReport,
    path: Sequence[str],
    depth: int,
) -> str:
    if depth >= depth_limit:
        report.unresolved.append(
            Unresolved(target.macro, target.describe(), _too_deep(depth_limit, path))
        )
        return ""

    page = await lookup(target)
    if page is None or page.storage is None:
        report.unresolved.append(
            Unresolved(target.macro, target.describe(), _not_found(target, page))
        )
        return ""
    if page.page_id in path:
        report.unresolved.append(Unresolved(target.macro, target.describe(), _cycle(page, path)))
        return ""

    body = excerpt_of_storage(page.storage) if target.wants_excerpt else page.storage
    if body is None:
        report.unresolved.append(Unresolved(target.macro, target.describe(), _no_excerpt(page)))
        return ""

    report.included.append(page.page_id)
    return await resolve_storage(
        body,
        lookup=lookup,
        depth_limit=depth_limit,
        report=report,
        path=(*path, page.page_id),
        depth=depth + 1,
    )


def excerpt_of_storage(body: str) -> str | None:
    """The markup inside a page's ``excerpt`` macro, or ``None`` when it defines none."""
    for macro in storage_macros(body):
        if macro.name == EXCERPT_MACRO and macro.body is not None:
            start, end = macro.body
            return body[start:end]
    return None


def storage_macros(body: str) -> list[StorageMacro]:
    """Every ``ac:structured-macro`` element in ``body``, at any nesting, with its exact span.

    Nested ones are included because an ``include`` inside an ``info`` panel is still an
    include, and a scan that reported only the outermost element would leave it unexpanded —
    which is the failure this module exists to prevent, arriving through the scanner instead of
    through the resolver. Two include spans can never overlap, because an include macro has no
    body to contain another one, so splicing them all is well defined.
    """
    scanner = _StorageScanner(body)
    scanner.feed(body)
    scanner.close()
    return scanner.macros


_MACRO_TAG = "ac:structured-macro"
_PARAMETER_TAG = "ac:parameter"
_RICH_TEXT_TAG = "ac:rich-text-body"
_PAGE_TAG = "ri:page"


@dataclass(slots=True)
class _Open:
    """A macro element being read."""

    name: str
    start: int
    params: dict[str, str] = field(default_factory=dict[str, str])
    body_start: int | None = None
    body_end: int | None = None
    depth: int = 0
    """How many ``ac:rich-text-body`` elements are open inside it, so the first one closing is
    matched with the first one opening rather than with a nested macro's."""


class _StorageScanner(HTMLParser):
    """Locates macro elements in storage format by parsing it, never by matching text.

    Storage format is XHTML with ``ac:`` and ``ri:`` prefixes whose namespaces are declared
    nowhere in the fragment, so an XML parser refuses it outright and an HTML parser reads the
    prefixed names as ordinary tags — which is all this needs. It reports *positions*, and the
    caller splices the original string, so nothing is re-serialised.
    """

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self._source = source
        self._line_starts = _line_starts(source)
        self._stack: list[_Open] = []
        self._parameter: str | None = None
        self.macros: list[StorageMacro] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._line_starts[line - 1] + column

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        if tag == _MACRO_TAG:
            self._stack.append(_Open(name=values.get("ac:name", ""), start=self._offset()))
            return
        if not self._stack:
            return
        current = self._stack[-1]
        if tag == _PARAMETER_TAG:
            self._parameter = values.get("ac:name", "").strip().lower()
            current.params.setdefault(self._parameter, "")
        elif tag == _PAGE_TAG:
            self._page_reference(current, values)
        elif tag == _RICH_TEXT_TAG:
            if current.depth == 0 and current.body_start is None:
                start = self._offset()
                raw = self.get_starttag_text() or ""
                current.body_start = start + len(raw)
            current.depth += 1

    @override
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        if tag == _PAGE_TAG and self._stack:
            self._page_reference(self._stack[-1], values)
        elif tag == _MACRO_TAG:
            raw = self.get_starttag_text() or ""
            start = self._offset()
            self.macros.append(
                StorageMacro(name=values.get("ac:name", ""), start=start, end=start + len(raw))
            )

    def _page_reference(self, current: _Open, values: Mapping[str, str]) -> None:
        """A ``ri:page`` inside a parameter names the target as a link rather than as text."""
        title = values.get("ri:content-title", "").strip()
        space = values.get("ri:space-key", "").strip()
        if title and self._parameter is not None:
            current.params[self._parameter] = title
        if space:
            current.params.setdefault("spacekey", space)

    @override
    def handle_data(self, data: str) -> None:
        if self._parameter is None or not self._stack:
            return
        current = self._stack[-1]
        current.params[self._parameter] = (current.params.get(self._parameter, "") + data).strip()

    @override
    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        current = self._stack[-1]
        if tag == _PARAMETER_TAG:
            self._parameter = None
            return
        if tag == _RICH_TEXT_TAG:
            current.depth -= 1
            if current.depth == 0 and current.body_end is None:
                current.body_end = self._offset()
            return
        if tag != _MACRO_TAG:
            return
        self._stack.pop()
        closed = self._source.find(">", self._offset())
        end = len(self._source) if closed < 0 else closed + 1
        body = (
            (current.body_start, current.body_end)
            if current.body_start is not None and current.body_end is not None
            else None
        )
        self.macros.append(
            StorageMacro(
                name=current.name, start=current.start, end=end, params=current.params, body=body
            )
        )


def _line_starts(source: str) -> list[int]:
    """Absolute offset of the start of each line, so ``getpos()`` can be turned into one.

    Found with :meth:`str.find` rather than by looping over characters: a page body is
    routinely tens of thousands of characters and this runs once per macro scan, which is up
    to three times per fetched page.
    """
    starts = [0]
    index = source.find("\n")
    while index != -1:
        starts.append(index + 1)
        index = source.find("\n", index + 1)
    return starts


# --- reporting without resolving -----------------------------------------------------------


def find_adf_macros(document: Mapping[str, object]) -> list[MacroTarget]:
    """Every include macro in a document, without expanding any of them."""
    found = (adf_macro_target(node) for node in _walk_adf(_content_of(document)))
    return [target for target in found if target is not None]


def find_storage_macros(body: str) -> list[MacroTarget]:
    """Every include macro in a storage-format body, without expanding any of them."""
    return [macro.target for macro in storage_macros(body) if macro.name in INCLUDE_MACROS]


def unresolved_because(targets: Sequence[MacroTarget], reason: str) -> list[Unresolved]:
    """Record macros nobody tried to expand.

    Turning resolution off is a legitimate choice — a corpus of pages that each include a
    dozen others may prefer them separate — but it must not be a *quiet* one. The content is
    still missing from the chunk while appearing present in the UI; the only difference is
    that this time somebody asked for it.
    """
    return [Unresolved(target.macro, target.describe(), reason) for target in targets]


# --- reasons -------------------------------------------------------------------------------


def _too_deep(limit: int, path: Sequence[str]) -> str:
    return (
        f"macro expansion reached the depth limit of {limit} "
        f"(pages already expanded: {' > '.join(path) or 'none'}). The content is on the page "
        f"it names; raise macro_depth if this nesting is deliberate."
    )


def _cycle(page: IncludedPage, path: Sequence[str]) -> str:
    return (
        f"page {page.page_id} ({page.title!r}) is already being expanded on this path "
        f"({' > '.join(path)}), so including it again would not terminate"
    )


def _not_found(target: MacroTarget, page: IncludedPage | None) -> str:
    if page is None:
        return (
            f"no page matching {target.describe()} was found, or this account cannot see it. "
            f"Confluence shows a macro error in the same case."
        )
    return f"page {page.page_id} was found but returned no body in the format being read"


def _no_excerpt(page: IncludedPage) -> str:
    return (
        f"page {page.page_id} ({page.title!r}) defines no excerpt macro, so an excerpt-include "
        f"of it renders nothing"
    )
