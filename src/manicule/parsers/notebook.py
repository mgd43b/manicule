"""Jupyter notebooks: the heading tree says where, the cell id says exactly where.

A notebook is a list of cells, and ``HeadingAnchor`` is the right shape for it because the
markdown cells carry a real heading hierarchy — ``#`` in a markdown cell is markdown, not a
line that happens to start with a hash. The fragment is the nbformat cell ``id``, prefixed
``cell-``, so a citation addresses one cell rather than a section that may hold twenty.

**Cell ids arrived in nbformat 4.5.** Below that the fragment is ``None`` and the heading path
is the only address available; where that path repeats, the block is
:class:`~manicule.core.anchors.Unlocated` and the reason says that saving the notebook as 4.5
or later fixes it, because that is a thing the reader can actually do
(``docs/parsing.md`` §2.5). This parser never converts a notebook to reach 4.5: conversion
*generates* ids, which would mint an address that is not in the file and would differ between
two machines that ran it.

**A cell id is not on its own a unique address here.** One markdown cell can open two
sections, and both carry the same cell id. So a block is located by ``(path, fragment)``
together and :meth:`NotebookParser.resolve` matches both — which makes the anchor tighter than
the cell, not looser: resolving the first section returns the first section.

**Outputs are content.** ``text/plain`` results and stream output are what a reader sees under
the code and frequently the only statement of the result, so they are emitted as a ``prose``
block on the same cell. An output that is only an image contributes no text — with optical
character recognition out of scope there is nothing to index — and is counted in
``metadata.image_outputs`` rather than becoming an empty block. Nothing here reads a code
comment as a heading.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TypeGuard

from nbformat import NBFormatError
from nbformat import reader as nbreader
from nbformat.reader import NotJSONError
from nbformat.validator import ValidationError

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import HeadingStack, ParserProfile, decode
from manicule.parsers.config import (
    NOTEBOOK_MEDIA_TYPE,
    NOTEBOOK_MEDIA_TYPES,
    NotebookConfig,
)

__all__ = ["NOTEBOOK_MEDIA_TYPE", "NOTEBOOK_MEDIA_TYPES", "NotebookConfig", "NotebookParser"]

CELL_ID_MINOR = 5
"""The nbformat minor version that introduced cell ids. Below it, fragments are ``None``."""

_SUPPORTED_MAJOR = 4

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_FENCE = re.compile(r"^\s*(?:```|~~~)")


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    """Whether a decoded JSON value is an object. Its keys are strings by construction."""
    return isinstance(value, dict)


def _is_array(value: object) -> TypeGuard[list[object]]:
    """Whether a decoded JSON value is an array."""
    return isinstance(value, list)


def _mapping(value: object) -> dict[str, object]:
    """A JSON object, or an empty one — a malformed cell is skipped, never guessed at."""
    return value if _is_object(value) else {}


def _array(value: object) -> list[object]:
    return value if _is_array(value) else []


def _string(value: object) -> str:
    """A notebook string field, which nbformat allows to be a string or a list of lines."""
    if isinstance(value, str):
        return value
    if _is_array(value):
        return "".join(item for item in value if isinstance(item, str))
    return ""


def _whole(value: object, default: int) -> int:
    """An integer JSON field. ``bool`` is excluded because it is an ``int`` and is not one."""
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


@dataclass(frozen=True, slots=True)
class _Heading:
    """An ATX heading inside a markdown cell. It opens a section rather than sitting in one."""

    level: int
    text: str


@dataclass(frozen=True, slots=True)
class _Item:
    """One block-to-be: everything but the anchor, which belongs to its group."""

    kind: BlockKind
    text: str
    lang: str | None = None
    metadata: Metadata | None = None

    @property
    def extras(self) -> Metadata:
        return dict(self.metadata) if self.metadata else {}


@dataclass(frozen=True, slots=True)
class _Placed:
    """An item with the address it inherited from the cells before it."""

    path: tuple[str, ...]
    fragment: str | None
    heading_path: tuple[str, ...]
    item: _Item


@dataclass(frozen=True, slots=True)
class _Group:
    """Everything one address covers: a run of items sharing a path and a cell id."""

    path: tuple[str, ...]
    fragment: str | None
    heading_path: tuple[str, ...]
    items: tuple[_Item, ...]

    @property
    def key(self) -> tuple[tuple[str, ...], str | None]:
        return self.path, self.fragment

    @property
    def text(self) -> str:
        """The group as one string. What :meth:`NotebookParser.resolve` returns."""
        return "\n".join(item.text for item in self.items)


def _markdown_elements(source: str) -> list[_Heading | str]:
    """Split a markdown cell into headings and the prose between them.

    Fenced blocks are tracked because ``# not a heading`` inside a fence is a comment in
    whatever language the fence holds. Reading it as a heading would put it in the path of
    every cell below, and the path reaches the embedder through the breadcrumb.
    """
    elements: list[_Heading | str] = []
    prose: list[str] = []
    fenced = False

    def flush() -> None:
        text = "\n".join(prose).strip()
        prose.clear()
        if text:
            elements.append(text)

    for line in source.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            prose.append(line)
            continue
        match = None if fenced else _ATX_HEADING.match(line)
        if match is None:
            prose.append(line)
            continue
        flush()
        text = _CLOSING_HASHES.sub("", match.group(2)).strip()
        if text:
            elements.append(_Heading(level=len(match.group(1)), text=text))
    flush()
    return elements


def _render_outputs(outputs: Iterable[object]) -> tuple[str, int]:
    """The text a cell's outputs contribute, and how many outputs were images only."""
    lines: list[str] = []
    images = 0
    for entry in outputs:
        output = _mapping(entry)
        kind = _string(output.get("output_type"))
        if kind == "stream":
            text = _string(output.get("text")).rstrip("\n")
            if text:
                lines.append(text)
            continue
        if kind in {"execute_result", "display_data"}:
            data = _mapping(output.get("data"))
            plain = _string(data.get("text/plain")).rstrip("\n")
            if plain:
                lines.append(plain)
            elif any(key.startswith("image/") for key in data):
                images += 1
            continue
        if kind == "error":
            name = _string(output.get("ename"))
            value = _string(output.get("evalue"))
            summary = f"{name}: {value}".strip(": ")
            if summary:
                lines.append(summary)
            traceback = "\n".join(
                item for item in (_string(line) for line in _array(output.get("traceback"))) if item
            ).rstrip("\n")
            if traceback:
                lines.append(traceback)
    return "\n".join(lines), images


def _language(notebook: Mapping[str, object]) -> str | None:
    """The notebook's code language, or ``None`` when it declares none.

    ``None`` rather than "python": a notebook that does not say is not evidence that it is, and
    a wrong language tag on a code block sends it to the wrong syntax highlighter and the wrong
    grammar.
    """
    metadata = _mapping(notebook.get("metadata"))
    declared = _string(_mapping(metadata.get("language_info")).get("name")).strip()
    if declared:
        return declared
    return _string(_mapping(metadata.get("kernelspec")).get("language")).strip() or None


def _title(raw: RawDocument, notebook: Mapping[str, object]) -> str:
    """The document's title: what the connector reported, else what the notebook declares."""
    declared = raw.metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return _string(_mapping(notebook.get("metadata")).get("title")).strip()


def _placed_items(
    notebook: Mapping[str, object], minor: int, config: NotebookConfig, title: str
) -> list[_Placed]:
    """Walk the cells in order, carrying the heading path forward as markdown cells set it."""
    stack = HeadingStack()
    language = _language(notebook)
    path: tuple[str, ...] = (title,) if title else ()
    heading_path: tuple[str, ...] = ()
    placed: list[_Placed] = []

    for entry in _array(notebook.get("cells")):
        cell = _mapping(entry)
        fragment = _fragment(cell, minor)
        source = _string(cell.get("source"))
        cell_type = _string(cell.get("cell_type"))

        if cell_type == "markdown":
            for element in _markdown_elements(source):
                if isinstance(element, _Heading):
                    heading_path = stack.push(element.level, element.text)
                    path = heading_path
                    item = _Item(
                        kind=BlockKind.HEADING,
                        text=element.text,
                        metadata={"level": element.level},
                    )
                else:
                    item = _Item(kind=BlockKind.PROSE, text=element)
                placed.append(_Placed(path, fragment, heading_path, item))
            continue

        if cell_type == "code":
            code = source.rstrip("\n")
            outputs, images = (
                _render_outputs(_array(cell.get("outputs"))) if config.include_outputs else ("", 0)
            )
            if code.strip():
                metadata: Metadata = {"cell_type": "code"}
                if images:
                    metadata["image_outputs"] = images
                placed.append(
                    _Placed(
                        path,
                        fragment,
                        heading_path,
                        _Item(kind=BlockKind.CODE, text=code, lang=language, metadata=metadata),
                    )
                )
            if outputs.strip():
                placed.append(
                    _Placed(
                        path,
                        fragment,
                        heading_path,
                        _Item(kind=BlockKind.PROSE, text=outputs, metadata={"cell_output": True}),
                    )
                )
            continue

        if cell_type == "raw" and config.include_raw_cells:
            text = source.strip()
            if text:
                placed.append(
                    _Placed(
                        path,
                        fragment,
                        heading_path,
                        _Item(kind=BlockKind.PROSE, text=text, metadata={"cell_type": "raw"}),
                    )
                )

    return placed


def _fragment(cell: Mapping[str, object], minor: int) -> str | None:
    """``cell-<id>``, or ``None`` for a notebook whose format has no ids to read."""
    if minor < CELL_ID_MINOR:
        return None
    cell_id = _string(cell.get("id")).strip()
    return f"cell-{cell_id}" if cell_id else None


def _grouped(placed: Iterable[_Placed]) -> list[_Group]:
    """Collapse consecutive items that share an address into one group."""
    groups: list[_Group] = []
    for entry in placed:
        if groups and groups[-1].key == (entry.path, entry.fragment):
            groups[-1] = replace(groups[-1], items=(*groups[-1].items, entry.item))
            continue
        groups.append(_Group(entry.path, entry.fragment, entry.heading_path, (entry.item,)))
    return groups


def _anchor_for(
    group: _Group, keys: Mapping[tuple[tuple[str, ...], str | None], int], version: tuple[int, int]
) -> Anchor:
    """The anchor for a group, or :class:`Unlocated` when nothing addresses it uniquely."""
    if not group.path:
        return Unlocated(
            reason="this cell precedes the notebook's first markdown heading and the notebook "
            "declares no title, so there is no section path to cite. Add a heading cell above "
            "it, or give the notebook a title in its metadata"
        )
    if keys[group.key] == 1:
        return HeadingAnchor(path=group.path, fragment=group.fragment)
    rendered = " > ".join(group.path)
    if group.fragment is None:
        major, minor = version
        return Unlocated(
            reason=f"this notebook is nbformat {major}.{minor}, which predates cell ids, so the "
            f"heading path {rendered!r} is the only address available — and it names "
            f"{keys[group.key]} places. Saving the notebook as nbformat 4.5 or later gives every "
            f"cell an id and makes each one citable"
        )
    return Unlocated(
        reason=f"{group.fragment!r} is the id of {keys[group.key]} cells under {rendered!r}, and "
        f"a cell id has to be unique to address anything. Re-save the notebook from Jupyter, "
        f"which assigns fresh ids"
    )


class NotebookParser:
    """Parses a `.ipynb` into markdown, code and output blocks anchored by heading and cell.

    ``max_unlocated_ratio`` is 0.05 (``docs/parsing.md`` §3.4), and both things it pays for are
    properties of the file rather than of this parser: a notebook below nbformat 4.5 whose
    heading path repeats, and cells before the first heading in a notebook with no title.
    """

    media_types = NOTEBOOK_MEDIA_TYPES
    profile = ParserProfile(name="notebook", max_unlocated_ratio=0.05, max_pagelevel_ratio=None)

    def __init__(self, config: NotebookConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield blocks cell by cell, in notebook order, outputs after their code."""
        groups, version = self._read(raw)
        keys = Counter(group.key for group in groups)
        for group in groups:
            anchor = _anchor_for(group, keys, version)
            for item in group.items:
                yield ParsedBlock(
                    kind=item.kind,
                    text=item.text,
                    anchor=anchor,
                    heading_path=group.heading_path,
                    lang=item.lang,
                    metadata={
                        **item.extras,
                        "nbformat": f"{version[0]}.{version[1]}",
                    },
                )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the cells ``anchor`` addresses, re-derived from ``raw``.

        Both halves of the address are matched. A cell id alone would resolve a two-heading
        markdown cell to the whole cell, which would make the citation of its first section
        quote its second as well.
        """
        if not isinstance(anchor, HeadingAnchor):
            return None
        groups, _ = self._read(raw)
        keys = Counter(group.key for group in groups)
        for group in groups:
            if group.key == (anchor.path, anchor.fragment) and keys[group.key] == 1:
                return group.text
        return None

    def _read(self, raw: RawDocument) -> tuple[list[_Group], tuple[int, int]]:
        notebook, version = _open(raw)
        placed = _placed_items(notebook, version[1], self._config, _title(raw, notebook))
        return _grouped(placed), version


def _read_json(text: str) -> object:
    """Parse notebook JSON with nbformat's conversion-free reader.

    Wrapped so the suppression below has a home. nbformat ships ``py.typed`` and still leaves
    ``reader.reads`` unannotated, so the checker is right about upstream rather than about this
    code; the result is narrowed to JSON types by the helpers above before anything reads it.
    """
    return nbreader.reads(text)


def _open(raw: RawDocument) -> tuple[dict[str, object], tuple[int, int]]:
    """Read the notebook and its format version, declining anything that is not one.

    ``nbformat.reader`` is used rather than ``nbformat.read``, which validates and converts.
    Conversion is the problem: upgrading a 4.4 notebook to 4.5 generates the cell ids the file
    does not have, so every fragment would be an address invented at parse time — and a
    different one on the next run.
    """
    text = decode(raw)
    try:
        document = _read_json(text)
    except (NotJSONError, ValidationError, NBFormatError, AttributeError, ValueError) as exc:
        msg = (
            f"{raw.uri}: not a readable notebook ({type(exc).__name__}: {exc}). Expected "
            f"nbformat JSON with a 'cells' array. Open and re-save it in Jupyter, or route this "
            f"media type to a different parser"
        )
        raise ParseError(msg) from exc

    notebook = _mapping(document)
    major = _whole(notebook.get("nbformat"), 0)
    minor = _whole(notebook.get("nbformat_minor"), 0)
    if major != _SUPPORTED_MAJOR:
        msg = (
            f"{raw.uri}: nbformat {major}.{minor} is not readable here; this parser reads "
            f"nbformat {_SUPPORTED_MAJOR}. Upgrade the file with "
            f"'jupyter nbconvert --to notebook --inplace' and index it again — converting it "
            f"here would generate cell ids the file does not contain, so every citation into it "
            f"would point at an address invented on the machine that ran the conversion"
        )
        raise ParseError(msg)
    return notebook, (major, minor)
