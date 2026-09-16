"""A minimal compound-file writer, so ``.msg`` fixtures can be generated rather than committed.

``olefile`` reads compound files and does not write them, and every other writer is a
dependency this project would have to justify for a test corpus. So this writes the subset an
Outlook ``.msg`` actually uses: one root storage, a flat set of storages under it, and streams
that are all small enough to live in the mini stream.

**Small enough is not an assumption, it is enforced.** A reader decides where a stream lives
from its recorded size against the 4096-byte cutoff in the header, so a fixture that grew past
it would be looked for in the wrong place and read as empty. :func:`build` refuses instead.

The directory is written as a flat chain of black nodes rather than a balanced red-black tree.
That is what the format permits a reader to accept and what ``olefile`` does accept; a fixture
generator is not the place to implement tree balancing, and the round-trip test beside these
fixtures is what says the output is readable rather than merely plausible.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from itertools import pairwise

SECTOR = 512
MINI_SECTOR = 64
MINI_CUTOFF = 4096

_FREE = 0xFFFFFFFF
_END_OF_CHAIN = 0xFFFFFFFE
_FAT_SECTOR = 0xFFFFFFFD
_NO_STREAM = 0xFFFFFFFF

_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass
class Storage:
    """One storage and the streams directly inside it."""

    name: str
    streams: dict[str, bytes] = field(default_factory=dict[str, bytes])


class StreamTooLargeError(ValueError):
    """A fixture stream is at or past the mini-stream cutoff."""


def build(root_streams: dict[str, bytes], storages: list[Storage]) -> bytes:
    """A compound file holding ``root_streams`` and ``storages``, in that order."""
    entries: list[_Entry] = [_Entry(name="Root Entry", kind=5)]
    children: list[int] = []
    mini: list[bytes] = []
    offset = 0

    def place(data: bytes) -> tuple[int, int]:
        nonlocal offset
        if len(data) >= MINI_CUTOFF:
            msg = f"stream of {len(data)} bytes is at or past the {MINI_CUTOFF}-byte cutoff"
            raise StreamTooLargeError(msg)
        start = offset // MINI_SECTOR
        padded = data + b"\x00" * (-len(data) % MINI_SECTOR)
        mini.append(padded)
        offset += len(padded)
        return start, len(data)

    for name, data in root_streams.items():
        start, size = place(data)
        children.append(len(entries))
        entries.append(_Entry(name=name, kind=2, start=start, size=size))
    for storage in storages:
        inner: list[int] = []
        index = len(entries)
        entries.append(_Entry(name=storage.name, kind=1))
        for name, data in storage.streams.items():
            start, size = place(data)
            inner.append(len(entries))
            entries.append(_Entry(name=name, kind=2, start=start, size=size))
        entries[index].child = _chain(entries, inner)
        children.append(index)
    entries[0].child = _chain(entries, children)

    mini_stream = b"".join(mini)
    return _assemble(entries, mini_stream)


def _chain(entries: list[_Entry], indexes: list[int]) -> int:
    """Link siblings as a right-leaning chain and return the first, or ``_NO_STREAM``."""
    if not indexes:
        return _NO_STREAM
    for left, right in pairwise(indexes):
        entries[left].right = right
    return indexes[0]


@dataclass
class _Entry:
    name: str
    kind: int
    start: int = _END_OF_CHAIN
    size: int = 0
    child: int = _NO_STREAM
    left: int = _NO_STREAM
    right: int = _NO_STREAM

    def pack(self) -> bytes:
        encoded = self.name.encode("utf-16-le") + b"\x00\x00"
        return struct.pack(
            "<64sHBBIII16sIQQIII",
            encoded.ljust(64, b"\x00"),
            len(encoded),
            self.kind,
            1,  # black, because the directory here is a chain rather than a balanced tree
            self.left,
            self.right,
            self.child,
            b"\x00" * 16,
            0,
            0,
            0,
            self.start,
            self.size,
            0,
        )


def _sectors(data: bytes, size: int) -> list[bytes]:
    padded = data + b"\x00" * (-len(data) % size)
    return [padded[at : at + size] for at in range(0, len(padded), size)]


def _assemble(entries: list[_Entry], mini_stream: bytes) -> bytes:
    """Lay out the mini stream, the mini FAT, the directory and the FAT that chains them."""
    mini_sectors = _sectors(mini_stream, SECTOR)
    entries[0].start = 0 if mini_sectors else _END_OF_CHAIN
    entries[0].size = len(mini_stream)

    used = len(mini_stream) // MINI_SECTOR
    minifat = b"".join(
        struct.pack("<I", _END_OF_CHAIN if at + 1 == used else at + 1) for at in range(used)
    )
    minifat_sectors = _sectors(minifat, SECTOR) or [b"\xff" * SECTOR]
    directory = b"".join(entry.pack() for entry in entries)
    directory_sectors = _sectors(directory, SECTOR)

    layout = [
        ("mini", len(mini_sectors)),
        ("minifat", len(minifat_sectors)),
        ("directory", len(directory_sectors)),
    ]
    starts: dict[str, int] = {}
    fat = bytearray()
    at = 0
    for name, count in layout:
        starts[name] = at if count else _END_OF_CHAIN
        for index in range(count):
            last = index + 1 == count
            fat += struct.pack("<I", _END_OF_CHAIN if last else at + index + 1)
        at += count
    fat_sector = at
    fat += struct.pack("<I", _FAT_SECTOR)
    fat += b"\xff" * (SECTOR - len(fat) % SECTOR) if len(fat) % SECTOR else b""

    header = bytearray(b"\x00" * SECTOR)
    header[0:8] = _SIGNATURE
    struct.pack_into("<HHHH", header, 24, 0x003E, 3, 0xFFFE, 9)
    struct.pack_into("<H", header, 32, 6)  # mini sector shift: 2**6 == 64
    struct.pack_into("<I", header, 40, len(directory_sectors))
    struct.pack_into("<I", header, 44, 1)  # one FAT sector
    struct.pack_into("<I", header, 48, starts["directory"])
    struct.pack_into("<I", header, 56, MINI_CUTOFF)
    struct.pack_into("<I", header, 60, starts["minifat"])
    struct.pack_into("<I", header, 64, len(minifat_sectors))
    struct.pack_into("<I", header, 68, _END_OF_CHAIN)  # no DIFAT beyond the header
    struct.pack_into("<I", header, 72, 0)
    struct.pack_into("<I", header, 76, fat_sector)
    for slot in range(1, 109):
        struct.pack_into("<I", header, 76 + slot * 4, _FREE)

    body = b"".join((*mini_sectors, *minifat_sectors, *directory_sectors, bytes(fat[:SECTOR])))
    return bytes(header) + body


__all__ = ["MINI_CUTOFF", "Storage", "StreamTooLargeError", "build"]
