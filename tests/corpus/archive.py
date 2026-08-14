"""Fixtures for the zip parser, including the ones nobody should commit.

Generation is what makes this corpus possible. A zip bomb and a hostile member name are
inspectable as twenty lines of code here, and are never stored in the repository at all
(``docs/parsing.md`` §3.5).

Two of these are built by editing the archive after :mod:`zipfile` has written it, because
they are archives that lie about themselves and no writer produces one on request: a member
whose header declares a size its stream does not honor, and a member flagged as encrypted.
The editing is done against the record offsets the format defines rather than by scanning for
byte patterns, so it cannot land in the middle of compressed data and quietly corrupt
something else.
"""

from __future__ import annotations

import io
import struct
import warnings
import zipfile
from pathlib import Path

BOMB_DECLARED_SIZE = 1024
"""What the bomb's header claims, and the whole reason the test is worth writing.

Every limit a header-only check could apply passes at this size. What stops the member is the
counter around the stream, which is the only thing that can.
"""

BOMB_REAL_SIZE = 4 * 1024 * 1024
"""What the bomb actually expands to, from a member of a few kilobytes."""

SYMLINK_MODE = 0o120777 << 16
ENCRYPTED_FLAG = 0x1

_MEMBERS = {
    "readme.txt": (
        "This archive holds three documents and a directory entry.\n"
        "Each member is indexed as a document of its own, with its own anchors.\n"
    ),
    "reports/january.txt": (
        "January: ingest ran on twenty-two days and reconciled on all of them.\n"
    ),
    "reports/february.md": (
        "# February\n\nThe watermark change landed and the resume path was exercised twice.\n"
    ),
}


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    (dest / "typical.zip").write_bytes(_typical())
    (dest / "nested.zip").write_bytes(_nested())
    (dest / "empty.zip").write_bytes(_empty())
    (dest / "wide.zip").write_bytes(_wide())
    (dest / "bomb.zip").write_bytes(_bomb())
    (dest / "traversal.zip").write_bytes(_traversal())
    (dest / "colliding.zip").write_bytes(_colliding())
    (dest / "symlink.zip").write_bytes(_symlink())
    (dest / "encrypted.zip").write_bytes(_encrypted())
    (dest / "ooxml.zip").write_bytes(_ooxml())
    (dest / "ebook.zip").write_bytes(_ebook())
    (dest / "inner.zip").write_bytes(_inner())
    (dest / "zero-bytes.zip").write_bytes(b"")


def _archive(entries: dict[str, bytes | str], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A zip with fixed timestamps, so two runs of the generator produce identical bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, content in entries.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 2, 3, 9, 14, 0))
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            data = content.encode("utf-8") if isinstance(content, str) else content
            archive.writestr(info, data)
    return buffer.getvalue()


def _typical() -> bytes:
    return _archive({"reports/": b"", **_MEMBERS})


def _inner() -> bytes:
    """The archive that goes inside another one, and stands alone as a cycle fixture."""
    return _archive({"inner-note.txt": "A document one level further in.\n"})


def _nested() -> bytes:
    return _archive(
        {
            "outer-note.txt": "A document at the top level of the outer archive.\n",
            "bundle/inner.zip": _inner(),
        }
    )


def _empty() -> bytes:
    return _archive({})


def _wide() -> bytes:
    """Many small honest members: the variant every byte limit passes."""
    return _archive(
        {
            f"notes/note-{number:03d}.txt": f"Note {number:03d} and nothing else.\n"
            for number in range(60)
        }
    )


def _bomb() -> bytes:
    """A member that declares a kilobyte and expands to megabytes.

    The declared size is rewritten in both the local header and the central directory, and the
    stored CRC is left describing the real content — which is what an attacker wants, because
    a member nobody can read is a member nobody indexes.
    """
    payload = b"\x00" * BOMB_REAL_SIZE
    data = _archive({"bomb.bin": payload})
    return _rewrite_size(data, "bomb.bin", BOMB_DECLARED_SIZE)


def _traversal() -> bytes:
    return _archive(
        {
            "safe.txt": "An ordinary member, so the archive is not entirely hostile.\n",
            "../escape.txt": "A member whose name climbs out of the archive root.\n",
        }
    )


def _colliding() -> bytes:
    """Three members that normalize to two names, plus two that normalize to none.

    A zip may hold two entries with the same name — appending to an archive produces exactly
    that — and ``a/b.txt`` and ``./a/b.txt`` normalize to one name as well. Storage reconciles
    members by ``source_id``, which is derived from the normalized name, so a collision is not
    an error anybody sees: the later member overwrites the earlier one and the archive quietly
    contributes fewer documents than it contains.

    Built by hand rather than through ``_archive`` because a dict cannot hold a duplicate key,
    which is the whole shape being tested.
    """
    entries = [
        ("reports/q1.txt", "The first quarter, written once.\n"),
        ("./reports/q1.txt", "The first quarter again, under a name that normalizes the same.\n"),
        ("reports/q1.txt", "The first quarter a third time, under a literally identical name.\n"),
        ("../escape-one.txt", "One member whose name climbs out of the root.\n"),
        ("../escape-two.txt", "Another, so a shared placeholder would collide.\n"),
    ]
    buffer = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # zipfile warns on a duplicate member name, and the suite turns warnings into errors.
        # A duplicate name is precisely what this fixture is, so the warning is suppressed here
        # rather than globally — an unexpected duplicate anywhere else must still fail.
        warnings.filterwarnings("ignore", message="Duplicate name", category=UserWarning)
        for name, body in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 2, 3, 9, 14, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    return buffer.getvalue()


def _symlink() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        ordinary = zipfile.ZipInfo("target.txt", date_time=(2026, 2, 3, 9, 14, 0))
        ordinary.external_attr = 0o644 << 16
        archive.writestr(ordinary, "The file a symlink member would point at.\n")
        link = zipfile.ZipInfo("link.txt", date_time=(2026, 2, 3, 9, 14, 0))
        link.external_attr = SYMLINK_MODE
        archive.writestr(link, "target.txt")
    return buffer.getvalue()


def _encrypted() -> bytes:
    """A member flagged as encrypted.

    The flag is set after writing because :mod:`zipfile` cannot produce an encrypted entry, and
    the flag is what a reader checks. What matters for the test is that the member is reported
    with a reason and the archive keeps going, which the flag alone exercises.
    """
    data = _archive(
        {
            "plain.txt": "A member anyone can read.\n",
            "secret.txt": "A member whose header says it is encrypted.\n",
        }
    )
    return _rewrite_flags(data, "secret.txt", ENCRYPTED_FLAG)


def _ooxml() -> bytes:
    """An Office file with a ``.zip`` extension: the realistic version of the §9.4 bug."""
    return _archive(
        {
            "[Content_Types].xml": '<?xml version="1.0"?><Types/>',
            "_rels/.rels": '<?xml version="1.0"?><Relationships/>',
            "word/document.xml": "<w:document><w:body><w:p/></w:body></w:document>",
        }
    )


def _ebook() -> bytes:
    """An ODF or EPUB container, told apart by an uncompressed ``mimetype`` member first."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        first = zipfile.ZipInfo("mimetype", date_time=(2026, 2, 3, 9, 14, 0))
        first.compress_type = zipfile.ZIP_STORED
        archive.writestr(first, "application/epub+zip")
        # Every remaining member goes in with an explicit ZipInfo too. Handed a bare name,
        # zipfile stamps the current clock — which two builds inside the same two-second
        # window agree on and two builds either side of one do not, so the corpus was
        # reproducible most of the time and that is the worst kind.
        for name, body in (
            ("META-INF/container.xml", '<?xml version="1.0"?><container/>'),
            ("OEBPS/chapter-1.xhtml", "<html><body><p>One.</p></body></html>"),
        ):
            entry = zipfile.ZipInfo(name, date_time=(2026, 2, 3, 9, 14, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, body)
    return buffer.getvalue()


# --- editing an archive after it is written ------------------------------------------------


def _rewrite_size(data: bytes, name: str, declared: int) -> bytes:
    """Rewrite a member's declared uncompressed size in both places the format records it."""
    return _rewrite(data, name, local_offset=22, central_offset=24, value=declared, width=4)


def _rewrite_flags(data: bytes, name: str, flags: int) -> bytes:
    """Set a member's general-purpose bit flags in both places the format records them."""
    return _rewrite(data, name, local_offset=6, central_offset=8, value=flags, width=2)


def _rewrite(
    data: bytes, name: str, *, local_offset: int, central_offset: int, value: int, width: int
) -> bytes:
    editable = bytearray(data)
    form = "<I" if width == 4 else "<H"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        header = archive.getinfo(name).header_offset
    struct.pack_into(form, editable, header + local_offset, value)
    struct.pack_into(form, editable, _central_record(data, name) + central_offset, value)
    return bytes(editable)


def _central_record(data: bytes, name: str) -> int:
    """Where a member's central directory record starts, walked rather than searched for.

    Searching for the ``PK\\x01\\x02`` signature would find it inside compressed data as often
    as not. The end-of-central-directory record says where the directory begins, and every
    record says how long it is, so walking is exact.
    """
    end = data.rindex(b"PK\x05\x06")
    offset = int.from_bytes(data[end + 16 : end + 20], "little")
    target = name.encode("utf-8")
    while data[offset : offset + 4] == b"PK\x01\x02":
        name_length = int.from_bytes(data[offset + 28 : offset + 30], "little")
        extra_length = int.from_bytes(data[offset + 30 : offset + 32], "little")
        comment_length = int.from_bytes(data[offset + 32 : offset + 34], "little")
        start = offset + 46
        if data[start : start + name_length] == target:
            return offset
        offset = start + name_length + extra_length + comment_length
    message = f"no central directory record for {name!r}"
    raise LookupError(message)
