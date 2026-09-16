"""draw.io diagrams, which arrive as attachments rather than as blocks.

A Confluence ``drawio`` macro stores its diagram as a page **attachment** and keeps only the
diagram's name and a revision in the macro body. So unlike the notations in
``docs/parsing.md`` §8.4 — which are code in a code block, reached by declaring a language —
this one is reached by claiming a media type and registering a parser. The extraction target
is the same, and that is the whole reason the two are separate work rather than one piece:
same goal, no shared mechanism.

**An ``mxfile`` is XML wrapping more XML.** Each ``<diagram>`` element holds an
``<mxGraphModel>``, and draw.io writes that inner document two ways: plain, or
``encodeURIComponent`` then raw-deflate then base64. Both spellings are read here and nothing
else is — a third encoding is refused rather than guessed at, because guessing wrong produces
a document full of plausible mojibake instead of a visible failure.

**A block's text is the decoded ``mxGraphModel``, and the relationships go in ``embed_text``.**
That is §8.4.2's rule applied unchanged: the lexical leg indexes ``chunks.text``, so a node id
or a style string stays searchable verbatim, and a citation into a diagram keeps quoting what
the file holds. The reading — labelled nodes and the edges between them — is produced by
:mod:`manicule.parsers.diagrams` under the ``mxfile`` language and installed by the same
``diagrams`` middleware that serves Graphviz and mermaid. One reader table, one rewrite, one
fingerprint.

**Two decompression bounds, because the payload is attacker-controlled twice.** An attachment
is a file from the corpus, so ``docs/parsing.md`` §9.3's ruling holds: a compressed payload
needs a bound on what it expands *to*, enforced while expanding, not a bound on what it
arrived as. Both the deflate stream inside a ``<diagram>`` and the ``zTXt`` chunk inside a
``.drawio.png`` are streamed against that ceiling.

**A ``DOCTYPE`` is refused outright.** ``xml.etree`` expands internal entities, which is the
billion-laughs attack with no bound available to stop it, and an ``mxfile`` has no legitimate
use for a document type declaration. Refusing the declaration is cheaper and more certain than
trying to bound its consequences.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
import struct
import zlib
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote
from xml.etree import ElementTree

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile, decode
from manicule.parsers.config import DRAWIO_MEDIA_TYPES, DrawioConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_PNG_KEYWORDS: Final = frozenset({b"mxfile", b"mxGraphModel"})
"""The ``tEXt``/``zTXt`` keywords draw.io writes its source under when exporting a ``.png``."""

MAX_CELLS: Final = 20_000
"""Most ``mxCell`` elements the reader in :mod:`manicule.parsers.diagrams` walks in one page.

A module constant rather than a :class:`~manicule.parsers.config.DrawioConfig` field because the
walk happens in the middleware, which never sees this parser's configuration. A setting that
appears to be in force and is not would be worse than a constant that says where it applies.
The reading is already bounded twice by statements and by characters, but both of those bound
the answer; this bounds the traversal, so a page with a million cells costs a bounded walk
rather than a bounded answer arrived at slowly."""

_CHUNK: Final = 64 * 1024
_MAX_PNG_CHUNKS: Final = 4096
"""A PNG is a chunk list, so a hostile one is an unbounded chunk list. Scanning stops here."""

_TAG = re.compile(r"<[^>]*>")
_WHITESPACE = re.compile(r"\s+")


class DrawioParser:
    """Reads an ``mxfile`` attachment into one block per diagram page."""

    media_types = DRAWIO_MEDIA_TYPES
    profile = ParserProfile(name="drawio", max_unlocated_ratio=0.00, max_pagelevel_ratio=None)
    """Zero, because draw.io writes an ``id`` on every ``<diagram>`` it saves.

    :func:`_page_anchor` can still return :class:`~manicule.core.anchors.Unlocated`, and must —
    a hand-written file with neither a name nor an id has nothing to address a page by. But that
    is not a shape an export produces, so a corpus of real diagrams that contained one would be
    describing a bug rather than a format."""

    def __init__(self, config: DrawioConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """One block per ``<diagram>``, holding that page's decoded ``mxGraphModel``."""
        for page in self._pages(raw):
            yield ParsedBlock(
                kind=BlockKind.CODE,
                text=page.model,
                anchor=page.anchor,
                heading_path=(page.name,) if page.name else (),
                lang="mxfile",
                metadata={"diagram_id": page.identifier} if page.identifier else {},
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Re-decode and return the page this anchor names.

        Deterministic rather than stored: the same bytes through the same decoder give the same
        page back, which is what makes a citation into a diagram resolvable at all — the file
        holds base64, and no offset into it addresses anything a reader would recognize.
        """
        if not isinstance(anchor, HeadingAnchor):
            return None
        for page in self._pages(raw):
            if page.anchor == anchor:
                return page.model
        return None

    def _pages(self, raw: RawDocument) -> Iterator[_Page]:
        source = _mxfile_source(raw, max_bytes=self._config.max_decompressed_bytes)
        root = _parse_xml(source)
        elements = [root] if root.tag == "diagram" else list(root.iter("diagram"))
        if not elements:
            msg = "no <diagram> element: this is not a draw.io mxfile"
            raise ParseError(msg)
        seen: set[str] = set()
        for ordinal, element in enumerate(elements[: self._config.max_diagrams], start=1):
            model = _diagram_model(element, max_bytes=self._config.max_decompressed_bytes)
            if model is None:
                continue
            name = (element.get("name") or "").strip()
            identifier = (element.get("id") or "").strip()
            yield _Page(
                name=name,
                identifier=identifier,
                model=model,
                anchor=_page_anchor(name, identifier, ordinal, seen),
            )


class _Page:
    """One ``<diagram>``: its name, its id, its decoded model and the anchor addressing it."""

    __slots__ = ("anchor", "identifier", "model", "name")

    def __init__(self, *, name: str, identifier: str, model: str, anchor: Anchor) -> None:
        self.name = name
        self.identifier = identifier
        self.model = model
        self.anchor = anchor


def _page_anchor(name: str, identifier: str, ordinal: int, seen: set[str]) -> Anchor:
    """The page's own name, else its own id, else nothing.

    An ordinal is deliberately not a fallback. A tab's position is not a name the file gave it,
    and ``docs/parsing.md`` §3 is explicit that an anchor is built from what the source states
    or is :class:`~manicule.core.anchors.Unlocated` with a reason. Two tabs sharing a name is
    the one case where position has to break the tie, and it says so in the fragment rather
    than in the path, so the first tab keeps the plain name a reader would cite.
    """
    label = name or identifier
    if not label:
        return Unlocated(reason="the diagram has neither a name nor an id to address it by")
    if label in seen:
        return HeadingAnchor(path=(label,), fragment=f"{ordinal}")
    seen.add(label)
    return HeadingAnchor(path=(label,))


def _mxfile_source(raw: RawDocument, *, max_bytes: int) -> str:
    """The ``<mxfile>`` XML, lifted out of a ``.drawio.png`` wrapper when there is one."""
    data = raw.as_bytes()
    if data.startswith(_PNG_SIGNATURE):
        embedded = _png_mxfile(data, max_bytes=max_bytes)
        if embedded is None:
            msg = "this PNG carries no embedded mxfile; it is a picture of a diagram, not one"
            raise ParseError(msg)
        return embedded
    return decode(raw)


def _png_mxfile(data: bytes, *, max_bytes: int) -> str | None:
    """The source draw.io embedded in a ``.drawio.png``, or ``None`` for a plain picture.

    A ``.drawio.png`` is a real PNG any viewer renders, with the diagram's own XML carried in a
    text chunk so that the export round-trips. Walking the chunk list is the only way to find
    it, and the walk is bounded because the list is attacker-controlled.
    """
    offset = len(_PNG_SIGNATURE)
    for _ in range(_MAX_PNG_CHUNKS):
        if offset + 8 > len(data):
            return None
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if len(body) != length:
            msg = "truncated PNG chunk: this file is not a readable draw.io export"
            raise ParseError(msg)
        if kind in {b"tEXt", b"zTXt"}:
            text = _png_text(kind, body, max_bytes=max_bytes)
            if text is not None:
                return text
        if kind == b"IEND":
            return None
        offset += 12 + length
    msg = f"more than {_MAX_PNG_CHUNKS} PNG chunks: refusing to keep scanning"
    raise ParseError(msg)


def _png_text(kind: bytes, body: bytes, *, max_bytes: int) -> str | None:
    """One ``tEXt``/``zTXt`` chunk's value, if its keyword is one draw.io writes."""
    keyword, separator, rest = body.partition(b"\x00")
    if not separator or keyword not in _PNG_KEYWORDS:
        return None
    if kind == b"tEXt":
        payload = rest
    else:
        if not rest or rest[0] != 0:
            msg = "zTXt chunk declares a compression method draw.io does not write"
            raise ParseError(msg)
        payload = _inflate(rest[1:], max_bytes=max_bytes, wbits=zlib.MAX_WBITS, what="zTXt chunk")
    return unquote(payload.decode("utf-8", errors="replace"))


def _diagram_model(element: ElementTree.Element, *, max_bytes: int) -> str | None:
    """One ``<diagram>``'s ``mxGraphModel``, in whichever of the two spellings it used."""
    nested = element.find("mxGraphModel")
    if nested is not None:
        return _serialize(nested)
    body = (element.text or "").strip()
    if not body:
        return None
    if body.startswith("<"):
        return body
    try:
        compressed = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = (
            "a <diagram> body is neither XML nor base64. draw.io writes one or the other; "
            f"refusing to guess at a third encoding ({exc})"
        )
        raise ParseError(msg) from exc
    inflated = _inflate(compressed, max_bytes=max_bytes, wbits=-zlib.MAX_WBITS, what="<diagram>")
    return unquote(inflated.decode("utf-8", errors="replace"))


def _inflate(data: bytes, *, max_bytes: int, wbits: int, what: str) -> bytes:
    """Expand ``data``, refusing the moment it passes ``max_bytes``.

    Bounded while expanding rather than checked afterwards, which is §9.3's ruling for zip
    members and is the same threat in a different container: a declared size is a field the
    attacker wrote, and by the time ``len()`` can be taken the memory is already spent.
    """
    engine = zlib.decompressobj(wbits)
    out = bytearray()
    pending = data
    while True:
        try:
            part = engine.decompress(pending, _CHUNK)
        except zlib.error as exc:
            msg = f"{what} does not hold a readable deflate stream ({exc})"
            raise ParseError(msg) from exc
        out += part
        if len(out) > max_bytes:
            msg = (
                f"{what} expands past the {max_bytes}-byte ceiling. Raise "
                f"parsers.drawio.max_decompressed_bytes to read it, or leave it refused."
            )
            raise ParseError(msg)
        if engine.eof:
            return bytes(out)
        pending = engine.unconsumed_tail
        if not pending:
            msg = f"{what} holds a truncated deflate stream"
            raise ParseError(msg)


def _parse_xml(source: str) -> ElementTree.Element:
    """Parse ``source``, refusing a document type declaration before expat can act on one."""
    if _has_doctype(source):
        msg = "an mxfile with a DOCTYPE is refused: entity expansion has no bound to stop it"
        raise ParseError(msg)
    try:
        return ElementTree.fromstring(source)  # noqa: S314 - refused above, and no DTD survives
    except ElementTree.ParseError as exc:
        msg = f"the mxfile is not well-formed XML ({exc})"
        raise ParseError(msg) from exc


def _has_doctype(source: str) -> bool:
    """Whether a document type declaration appears before the root element.

    A literal scan rather than an expat handler: the declaration can only be spelled one way
    and can only appear in the prolog, so finding it is exact, while reaching into
    :class:`xml.etree.ElementTree.XMLParser` for a handler means depending on an attribute the
    standard library does not document.
    """
    root = source.find("<mxfile")
    prolog = source if root < 0 else source[:root]
    return "<!DOCTYPE" in prolog


def _serialize(element: ElementTree.Element) -> str:
    return ElementTree.tostring(element, encoding="unicode")


def graph_source(source: str) -> ElementTree.Element | None:
    """A decoded ``mxGraphModel`` as a tree, or ``None`` when it cannot be read.

    The reader in :mod:`manicule.parsers.diagrams` walks a chunk's stored text, which arrived
    through this parser but need not have: a plugin can write ``Chunk.text``. So the same
    refusals apply on the way back in, and they are spelled once, here, rather than a second
    time beside the reader. ``None`` rather than a raise because a middleware that cannot read
    a diagram leaves the chunk alone — that is the module's contract for every other failure.
    """
    try:
        return _parse_xml(source)
    except ParseError:
        return None


def plain_text(value: str) -> str:
    """A cell label as a reader would see it, with draw.io's inline HTML removed.

    Labels are rich text: draw.io stores ``<b>Auth</b><br>Service`` for two styled lines, and
    an embedder given the markup reads the tags as content.
    """
    return _WHITESPACE.sub(" ", html.unescape(_TAG.sub(" ", value))).strip()


__all__: Sequence[str] = ["DrawioParser", "graph_source", "plain_text"]
