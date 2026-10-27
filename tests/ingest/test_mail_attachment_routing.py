"""Which attachments become text — through the real registry and the real runner.

The parser suites call ``parse()`` and pass. What decides whether an attachment is indexed is
the chain the *pipeline* resolves for the media type the mail parser assigned, so these drive
whole ``.eml`` messages through it.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Awaitable, Callable

import pytest

from manicule.core.content import Document, DocumentStatus, RawDocument
from manicule.core.fingerprints import ParseFingerprint
from manicule.ingest.reindex import re_parse_stale
from manicule.parsers import mail as mail_module
from manicule.parsers.versions import current_parse_fingerprints, parse_fingerprint
from tests.ingest import fakes
from tests.ingest.mail_routing_support import (
    docx_bytes,
    enclosed_message,
    message,
    real_pipeline,
    xlsx_bytes,
)


def tar_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


type Indexer = Callable[[bytes], Awaitable[tuple[dict[str, Document], dict[str, str]]]]


@pytest.fixture
def indexed() -> Indexer:
    """Ingest a message, returning ``(members, chunk text by member)``."""

    async def run(raw_message: bytes) -> tuple[dict[str, Document], dict[str, str]]:
        pipeline, store = await real_pipeline()
        await pipeline.ingest_raw(
            RawDocument(
                source_id="mail:m",
                uri="mail://m",
                media_type="message/rfc822",
                content=raw_message,
            ),
            source="mail",
        )
        members = {
            d.source_id.split("!/", 1)[1]: d
            for d in store.documents.values()
            if "!/" in d.source_id
        }
        text = {
            name: "\n".join(c.text for c in store.chunks.get(d.id, []))
            for name, d in members.items()
        }
        return members, text

    return run


@pytest.mark.parametrize(
    "case",
    [
        (
            "application/msword",
            docx_bytes("quarterly wordmarker"),
            "base64",
            "plan.docx",
            "wordmarker",
        ),
        (
            "application/vnd.ms-excel",
            xlsx_bytes("sheetmarker"),
            "base64",
            "plan.xlsx",
            "sheetmarker",
        ),
        ("application/octet-stream", docx_bytes("octetword"), "base64", "x.docx", "octetword"),
        ("application/octet-stream", xlsx_bytes("octetsheet"), "base64", "x.xlsx", "octetsheet"),
        (
            "application/octet-stream",
            b"plain octet marker\n",
            "base64",
            "notes.txt",
            "octet marker",
        ),
        ("application/octet-stream", b"sniffed text marker\n", "base64", "README", "sniffed text"),
        ("application/octet-stream", b"echo shellmarker\n", "base64", "run.sh", "shellmarker"),
        ("application/x-sh", b"echo shellalias\n", "7bit", "run2.sh", "shellalias"),
        (
            "text/markdown; charset=iso-8859-1",
            "# café latinmd\n".encode("latin-1"),
            "8bit",
            "n.md",
            "latinmd",
        ),
        (
            "text/plain; charset=utf-16",
            "utf16 marker\n".encode("utf-16"),
            "base64",
            "u.txt",
            "utf16 marker",
        ),
        (
            "text/plain; charset=us-ascii",
            "café mislabeled\n".encode(),
            "8bit",
            "m.txt",
            "mislabeled",
        ),
        ("application/json", b'{"jsonmarker": 1}\n', "7bit", "d.json", "jsonmarker"),
        (
            "application/ics",
            b"BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:icsmarker\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n",
            "7bit",
            "invite.ics",
            "icsmarker",
        ),
        (
            "application/octet-stream",
            b"BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:sniffedics\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n",
            "7bit",
            "event",
            "sniffedics",
        ),
    ],
)
async def test_an_attachment_is_indexed_whatever_the_client_called_it(
    indexed: Indexer, case: tuple[str, bytes, str, str, str]
) -> None:
    content_type, data, encoding, filename, marker = case
    members, text = await indexed(message((content_type, data, encoding, filename)))

    assert members[filename].status is DocumentStatus.INDEXED, members[filename].status_detail
    assert marker in text[filename]


async def test_an_enclosed_message_sent_in_base64_is_indexed(indexed: Indexer) -> None:
    raw = message(
        ("message/rfc822", enclosed_message("forwarded", "forwardmarker"), "base64", "fwd.eml")
    )
    members, text = await indexed(raw)

    assert members["fwd.eml"].status is DocumentStatus.INDEXED
    assert "forwardmarker" in text["fwd.eml"]


async def test_a_delivery_status_notification_keeps_every_recipient(indexed: Indexer) -> None:
    raw = (
        b"From: mailer@example.org\r\nSubject: bounce\r\nMIME-Version: 1.0\r\n"
        b"Content-Type: multipart/report; report-type=delivery-status; boundary=B\r\n\r\n"
        b"--B\r\nContent-Type: text/plain\r\n\r\nUndeliverable.\r\n"
        b"--B\r\nContent-Type: message/delivery-status\r\n\r\n"
        b"Reporting-MTA: dns; mx.example.org\r\n\r\n"
        b"Final-Recipient: rfc822; first@example.org\r\nStatus: 5.1.1\r\n\r\n"
        b"Final-Recipient: rfc822; second@example.org\r\nStatus: 5.2.2\r\n"
        b"--B--\r\n"
    )
    members, text = await indexed(raw)

    (name,) = members
    assert members[name].status is DocumentStatus.INDEXED
    assert "first@example.org" in text[name]
    assert "second@example.org" in text[name]


async def test_an_image_is_refused_before_its_bytes_are_kept(indexed: Indexer) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    members, _ = await indexed(
        message(
            ("image/png", png, "base64", "a.png"),
            ("application/octet-stream", png, "base64", "b.bin"),
        )
    )

    for name in ("a.png", "b.bin"):
        assert members[name].status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE
        assert "not indexed" in (members[name].status_detail or "")
        assert members[name].metadata["member_content_type"] == "image/png"


async def test_an_ole_document_declared_as_word_stays_unsupported(indexed: Indexer) -> None:
    """A legacy ``.doc`` must not be sent to the Word parser because its name says ``.docx``."""
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600
    members, _ = await indexed(message(("application/msword", ole, "base64", "old.docx")))

    assert members["old.docx"].status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_a_message_stored_before_the_fix_is_repaired_by_a_stale_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator step: ``document reindex --stale`` re-expands messages stored before #499.

    Messages are hash-unchanged, so a sync never re-expands them, and their unsupported members
    carry no parse lineage of their own. What reaches them is the ``email`` version bump, which
    makes the *message* stale; re-parsing it re-derives its members from the retained bytes.
    """
    store, blobs = fakes.MemoryIngestStore(), fakes.MemoryBlobs()
    raw = RawDocument(
        source_id="mail:m",
        uri="mail://m",
        media_type="message/rfc822",
        content=message(("application/msword", docx_bytes("repairedmarker"), "base64", "a.docx")),
    )

    def as_declared(declared: str, name: str, data: bytes) -> str:
        del name, data
        return declared

    def stored_by_the_old_parser(parser: str) -> ParseFingerprint | None:
        current = parse_fingerprint(parser)
        if current is None or parser != "email":
            return current
        return current.model_copy(update={"version": "4"})

    with monkeypatch.context() as old:
        old.setattr(mail_module, "resolve_member_media_type", as_declared)
        pipeline, _ = await real_pipeline(
            store=store, blobs=blobs, parse_fingerprints=stored_by_the_old_parser
        )
        await pipeline.ingest_raw(raw, source="mail")
    member = await store.find_document("mail", "mail:mail:m!/a.docx")
    assert member is not None
    assert member.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE, "the fixture is the old state"

    pipeline, _ = await real_pipeline(store=store, blobs=blobs)
    sweep = await re_parse_stale(
        store=store,
        pipeline=pipeline,
        blobs=blobs,
        parse_fingerprints=current_parse_fingerprints(),
    )

    assert sweep.reparsed == 1
    member = await store.find_document("mail", "mail:mail:m!/a.docx")
    assert member is not None
    assert member.status is DocumentStatus.INDEXED, "the sweep must not supersede its own repair"
    assert (
        member.media_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "repairedmarker" in "\n".join(c.text for c in store.chunks[member.id])


async def test_only_nameless_text_alternatives_that_are_documents_become_members(
    indexed: Indexer,
) -> None:
    raw = (
        b"From: pat@example.org\r\nSubject: invite\r\nMIME-Version: 1.0\r\n"
        b"Content-Type: multipart/alternative; boundary=B\r\n\r\n"
        b"--B\r\nContent-Type: text/plain\r\n\r\nPlease join.\r\n"
        b"--B\r\nContent-Type: text/x-amp-html\r\n\r\n<html amp4email></html>\r\n"
        b"--B\r\nContent-Type: text/calendar; method=REQUEST\r\n\r\n"
        b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n--B--\r\n"
    )
    members, _ = await indexed(raw)

    assert [m.media_type for m in members.values()] == ["text/calendar"]


@pytest.mark.parametrize(
    "case",
    [
        ("application/gzip", gzip.compress(b"gzmarker text\n"), "n.txt.gz", "n.txt", "gzmarker"),
        (
            "application/octet-stream",
            gzip.compress(b"# sniffgz\n"),
            "readme.md.gz",
            "readme.md",
            "sniffgz",
        ),
        (
            "application/x-tar",
            tar_bytes({"inner.md": b"# tarmarker\n"}),
            "bundle.tar",
            "inner.md",
            "tarmarker",
        ),
        (
            "application/gzip",
            gzip.compress(tar_bytes({"deep.txt": b"deepmarker\n"})),
            "bundle.tgz",
            "bundle.tar!/deep.txt",
            "deepmarker",
        ),
    ],
)
async def test_a_compressed_attachment_is_opened_and_its_content_indexed(
    indexed: Indexer, case: tuple[str, bytes, str, str, str]
) -> None:
    content_type, data, name, inner, marker = case
    members, text = await indexed(message((content_type, data, "base64", name)))

    assert members[name].status is DocumentStatus.CONTAINER
    (leaf,) = [key for key in members if key.endswith(f"!/{inner}")]
    assert members[leaf].status is DocumentStatus.INDEXED, members[leaf].status_detail
    assert marker in text[leaf]
