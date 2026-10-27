"""What a container member is, when what it declares is not enough.

A mail attachment arrives with a ``Content-Type`` the sending client chose, and clients choose
badly: a Word file as ``application/msword``, a PDF as ``application/x-pdf``, a spreadsheet as
``application/octet-stream`` because the client had no better word for it. ``docs/parsing.md``
§6.1 puts a type the source declared above a filename extension, and that stays true for every
type that is a claim. This module is the amendment for the types that are not: a **generic**
declaration (:data:`GENERIC`) says only that the sender did not classify the bytes, so the
filename extension is consulted, then the bytes, and the declaration is kept only when neither
says more.

**A name never routes to a parser that would fail rather than decline.** A ``.docx`` that is
really an OLE ``.doc`` must not reach the Word parser, so an extension proposes a type and the
bytes confirm it (:func:`sniff_media_type`). An extension the bytes do not confirm loses to
what the bytes say, and a file the sniffer cannot name keeps what it was declared as.

Standard library only, like :mod:`manicule.parsers.expansion`, which both container parsers
import.
"""

from __future__ import annotations

import io
import zipfile

from manicule.parsers.expansion import OCTET_STREAM, media_type_for

__all__ = [
    "ALIASES",
    "GENERIC",
    "UNINDEXED_MAIN_TYPES",
    "resolve_member_media_type",
    "sniff_media_type",
]

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

ALIASES: dict[str, str] = {
    "application/x-pdf": "application/pdf",
    "application/x-gzip": "application/gzip",
    "application/x-gunzip": "application/gzip",
    "application/ics": "text/calendar",
    "text/x-vcalendar": "text/calendar",
    "text/xml": "application/xml",
    "application/x-sh": "text/x-shellscript",
    "application/x-shellscript": "text/x-shellscript",
    "text/x-sh": "text/x-shellscript",
    "application/x-tar-gz": "application/gzip",
}
"""Spellings of a type the ecosystem uses for the type manicule routes on.

Resolved before anything else, because routing is by exact media type: ``application/x-pdf`` is
not a different kind of document from ``application/pdf``, and a chain registered for one would
otherwise never see the other."""

GENERIC: frozenset[str] = frozenset(
    {
        OCTET_STREAM,
        "binary/octet-stream",
        "application/x-download",
        "application/force-download",
        "application/unknown",
        # The legacy Office types are named here because mail clients attach OOXML under them
        # as often as they attach the binary formats they were registered for. No parser reads
        # the binary formats, so honoring the declaration would lose every OOXML file that
        # carries one; the bytes decide (an OLE file keeps the declared type and stays
        # unsupported, which is what it is).
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)
"""Declarations that say less than the filename or the bytes do."""

UNINDEXED_MAIN_TYPES: frozenset[str] = frozenset({"image", "video", "audio"})
"""Media types no parser reads and none is planned to (``docs/parsing.md`` §10.4)."""

_ZIP = b"PK\x03\x04"
_OLE = b"\xd0\xcf\x11\xe0"
_TNEF = b"\x22\x3e\x9f\x78"

_CONFIRMED = frozenset({DOCX, XLSX, PPTX, "application/zip", "application/pdf", "application/gzip"})


def sniff_media_type(data: bytes) -> str:
    """The media type the leading bytes establish, or :data:`OCTET_STREAM` when they do not.

    Only signatures that are unambiguous. Text is the one inference rather than a signature, and
    it is made last: bytes that decode as UTF-8 with no binary control characters are
    ``text/plain``, which is exactly what the plaintext parser would conclude, and anything it
    would decline is not offered to it.
    """
    result = _by_signature(data)
    if result is not None:
        return result
    from manicule.parsers.base import is_probably_text  # noqa: PLC0415 - see module docstring

    stripped = data[:8192].lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"BEGIN:VCALENDAR"):
        return "text/calendar"
    if stripped.startswith(b"<?xml"):
        return "application/xml"
    return "text/plain" if data and is_probably_text(data) else OCTET_STREAM


_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"%PDF-", "application/pdf"),
    (0, b"\x1f\x8b", "application/gzip"),
    (257, b"ustar", "application/x-tar"),
    (0, _OLE, OCTET_STREAM),
    (0, _TNEF, "application/vnd.ms-tnef"),
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    (0, b"II*\x00", "image/tiff"),
    (0, b"MM\x00*", "image/tiff"),
)
"""Fixed-offset signatures, in the order they are tried."""


def _by_signature(data: bytes) -> str | None:
    if data.startswith(_ZIP):
        return _zip_kind(data)
    for offset, signature, media_type in _SIGNATURES:
        if data[offset : offset + len(signature)] == signature:
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"mif1", b"msf1"}:
            return "image/heif"
        return "image/avif" if brand == b"avif" else "video/mp4"
    return None


def _zip_kind(data: bytes) -> str:
    """Which zip this is: an Office document by its fixed top-level directory, else a zip.

    Reads the central directory and nothing else, so a hostile archive costs a name list.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, ValueError, OSError):
        return OCTET_STREAM
    if "[Content_Types].xml" in names:
        for prefix, media_type in (("word/", DOCX), ("xl/", XLSX), ("ppt/", PPTX)):
            if any(name.startswith(prefix) for name in names):
                return media_type
    return "application/zip"


def resolve_member_media_type(declared: str, name: str, data: bytes) -> str:
    """The media type to route a container member under.

    1. A declared type with a spelling in :data:`ALIASES` is read as its canonical one.
    2. A declared type that is not :data:`GENERIC` is a claim and stands.
    3. Otherwise the filename extension proposes a type, and the bytes confirm it where a wrong
       answer would fail rather than decline (Office documents, PDFs, archives).
    4. With no usable extension the bytes decide.
    5. With neither, the declaration is kept.
    """
    declared = ALIASES.get(declared, declared)
    if declared not in GENERIC:
        return declared
    proposed = media_type_for(name)
    if proposed != OCTET_STREAM and proposed not in _CONFIRMED:
        return proposed
    sniffed = sniff_media_type(data)
    if proposed != OCTET_STREAM and sniffed == proposed:
        return proposed
    return sniffed if sniffed != OCTET_STREAM else declared
