"""Resolving what a container member is when its declared type says too little."""

from __future__ import annotations

import io
import zipfile

import pytest

from manicule.connectors.filesystem import (
    _MEDIA_TYPE_BY_SUFFIX as FILESYSTEM_SUFFIXES,  # pyright: ignore[reportPrivateUsage]
)
from manicule.parsers.expansion import OCTET_STREAM, media_type_for
from manicule.parsers.grammars import MEDIA_TYPES as GRAMMAR_MEDIA_TYPES
from manicule.parsers.mediatype import resolve_member_media_type, sniff_media_type
from tests.ingest.mail_routing_support import docx_bytes, xlsx_bytes

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512


def _case_id(value: object) -> str | None:
    """Deterministic ids: generated documents embed timestamps, and an id built from their bytes
    differs between xdist workers, which refuses to run when workers collect different tests."""
    if isinstance(value, bytes):
        return "bytes"
    return value if isinstance(value, str) else None


def _zip(*names: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "x")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.7\n", "application/pdf"),
        (docx_bytes("a"), DOCX),
        (xlsx_bytes("a"), XLSX),
        (_zip("a.txt"), "application/zip"),
        (b"#!/bin/sh\nexit 0\n" + _zip("a.txt"), "application/zip"),
        (b"#!/bin/sh\nexit 0\n" + _zip("[Content_Types].xml", "word/document.xml"), DOCX),
        (_zip("[Content_Types].xml", "other/x"), "application/zip"),
        (b"\x1f\x8b\x08\x00" + b"\x00" * 20, "application/gzip"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image/jpeg"),
        (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8, "image/heif"),
        (b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8, "video/mp4"),
        (b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n", "text/calendar"),
        (b"\xef\xbb\xbf<?xml version='1.0'?><a/>", "application/xml"),
        (b"just words\n", "text/plain"),
        (OLE, OCTET_STREAM),
        (b"\x00\x01\x02\x03" * 100, OCTET_STREAM),
        (b"", OCTET_STREAM),
    ],
    ids=_case_id,
)
def test_sniffing_names_only_what_the_bytes_establish(data: bytes, expected: str) -> None:
    assert sniff_media_type(data) == expected


@pytest.mark.parametrize(
    ("declared", "name", "data", "expected"),
    [
        # A declaration that is a claim stands, whatever the name says.
        ("application/pdf", "x.docx", b"%PDF-1.7", "application/pdf"),
        ("text/plain", "x.png", b"words", "text/plain"),
        # A spelling of a type is that type.
        ("application/x-pdf", "x.pdf", b"%PDF-1.7", "application/pdf"),
        ("application/ics", "i.ics", b"BEGIN:VCALENDAR", "text/calendar"),
        ("application/x-sh", "run.sh", b"echo", "text/x-shellscript"),
        ("text/xml", "a.xml", b"<a/>", "application/xml"),
        # A generic declaration yields to the extension, confirmed by the bytes.
        ("application/msword", "a.docx", docx_bytes("a"), DOCX),
        ("application/vnd.ms-excel", "a.xlsx", xlsx_bytes("a"), XLSX),
        (OCTET_STREAM, "a.pdf", b"%PDF-1.7", "application/pdf"),
        (OCTET_STREAM, "notes.txt", b"words", "text/plain"),
        # ...and to the bytes when there is no extension.
        (OCTET_STREAM, "README", b"words", "text/plain"),
        (OCTET_STREAM, "blob", docx_bytes("a"), DOCX),
        # An extension the bytes do not confirm loses to the bytes.
        (OCTET_STREAM, "a.docx", _zip("a.txt"), "application/zip"),
        ("application/msword", "a.docx", b"%PDF-1.7", "application/pdf"),
        # An OLE document is not OOXML however it is named: it keeps what it was declared.
        ("application/msword", "old.docx", OLE, "application/msword"),
        (OCTET_STREAM, "old.docx", OLE, OCTET_STREAM),
        # Nothing established anything.
        (OCTET_STREAM, "blob.bin", b"\x00\x01" * 100, OCTET_STREAM),
    ],
    ids=_case_id,
)
def test_resolution_order(declared: str, name: str, data: bytes, expected: str) -> None:
    assert resolve_member_media_type(declared, name, data) == expected


def test_the_shell_suffix_names_the_type_the_source_code_parser_claims() -> None:
    """The table is spelled out in ``expansion`` (no grammar import), so keep it honest."""
    assert media_type_for("run.sh") == GRAMMAR_MEDIA_TYPES["bash"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.docx", DOCX),
        ("a.XLSX", XLSX),
        ("a.log", "text/plain"),
        ("invite.ics", "text/calendar"),
        ("a.tgz", "application/gzip"),
        ("a.drawio.png", "application/vnd.jgraph.mxfile"),
        ("photo.png", OCTET_STREAM),
    ],
)
def test_the_suffix_table_names_what_the_filesystem_connector_does(
    name: str, expected: str
) -> None:
    assert media_type_for(name) == expected


def test_a_suffix_the_filesystem_connector_routes_routes_the_same_inside_a_container() -> None:
    """One answer to "what is ``report.docx``", whether it was a file or an attachment."""
    for suffix, expected in FILESYSTEM_SUFFIXES.items():
        assert media_type_for(f"x{suffix}") == expected, suffix
