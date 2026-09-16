"""Outlook ``.msg``, read as MAPI properties and handed to the ``.eml`` code path.

``.eml`` covers the common case; ``.msg`` is what actually lands in a corpus of exported mail.
It is Outlook's compound-file (OLE/CFBF) format, holding MAPI properties as named streams
rather than an RFC 5322 message — so reading it is a property reader rather than a library
call, and ``docs/parsing.md`` §10 sizes it as its own unit of work for that reason.

**A shim, not a second parser.** The transport headers and the body are read out and
reconstituted into an RFC 5322 message, which is then handed to
:class:`~manicule.parsers.mail.MailParser`. Anchors, the canonical-body rule, chunking and
round-trip behavior are therefore identical between the two formats *by construction* rather
than by two implementations agreeing — which is worth more than it costs, and is the whole
design. Attachments come along as ordinary MIME parts, so they become container members through
the same code that serves ``.eml``, under the same ``mail:`` scheme.

**Why a property reader rather than a dependency.** The maintained Python ``.msg`` library,
``extract-msg``, is GPL-3.0 and manicule is MIT. ``docs/parsing.md`` §12 records that decision
twice over — it was refused, admitted during the period this project was itself
GPL-3.0-or-later, and refused again — and ``tests/test_license_boundary.py`` enforces it rather
than trusting it. ``olefile`` is BSD-2-Clause and is the layer ``extract-msg`` itself sits on.

**The trap, and it is not an edge case.** ``PidTagTransportMessageHeaders`` is absent on any
message that never traversed a transport — drafts and everything in Sent Items, which is a large
share of what people export. There the headers are synthesized from the subject, the sender
properties and the recipient table, and the synthesized path is a required fixture (§3.5).

**Two spellings, read rather than deduced.** A string property is stored as UTF-16 under a
``001F`` stream or as 8-bit under ``001E``, signaled once by ``STORE_UNICODE_OK`` in
``PidTagStoreSupportMask``. Both stream names are looked for and whichever exists is read. That
is not guessing between them — only one is present — and it means a file that disagrees with its
own flag still reads, which a reader that trusted the flag would not.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator, Sequence
from email import message_from_string
from email.message import EmailMessage
from email.policy import SMTP
from io import BytesIO
from typing import TYPE_CHECKING, Final

from manicule.core.content import ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile
from manicule.parsers.config import MSG_MEDIA_TYPES, MsgConfig
from manicule.parsers.mail import MailParser

if TYPE_CHECKING:  # pragma: no cover - typing only
    from manicule.core.anchors import Anchor
    from manicule.parsers.expansion import MemberOutcome

_OLE_SIGNATURE: Final = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_TRANSPORT_HEADERS: Final = "007D"
_SUBJECT: Final = "0037"
_BODY: Final = "1000"
_HTML_BODY: Final = "1013"
_SENDER_NAME: Final = "0C1A"
_SENDER_EMAIL: Final = "0C1F"
_SENDER_SMTP: Final = "5D01"
_RECIPIENT_NAME: Final = "3001"
_RECIPIENT_EMAIL: Final = "3003"
_RECIPIENT_SMTP: Final = "39FE"
_ATTACH_LONG_NAME: Final = "3707"
_ATTACH_NAME: Final = "3704"
_ATTACH_DATA: Final = "3701"
_ATTACH_MIME_TAG: Final = "370E"

_UNICODE: Final = "001F"
_ANSI: Final = "001E"
_BINARY: Final = "0102"

_RECIPIENT_TYPE: Final = 0x0C150003
"""``PidTagRecipientType``, ``PT_LONG``: 1 is ``To``, 2 is ``Cc``, 3 is ``Bcc``."""

_PROPERTIES: Final = "__properties_version1.0"
"""Where fixed-width properties live, which is not beside the variable-width ones.

A ``PT_LONG`` has no ``__substg1.0_`` stream of its own — it is an eight-byte value inside one
properties stream per storage, behind a header whose length differs between the message and the
storages under it. A reader that looked for ``__substg1.0_0C150003`` finds nothing and concludes
every recipient is a ``To``, which is wrong on every real file and on no synthetic one.
"""

_PROPERTIES_HEADER: Final = 8
"""Reserved bytes before the first entry in a recipient's or attachment's properties stream.

The top-level message's stream has 32, because it carries the next-id and count fields as well.
Only the storages' streams are read here, so the shorter header is the one that matters.
"""

_PROPERTY_ENTRY: Final = 16
"""Four bytes of tag, four of flags, eight of value."""

_MAX_PROPERTIES_BYTES: Final = 1 << 20
"""Ceiling on one storage's properties stream, which holds fixed-width values and nothing else.

A megabyte is roughly sixty-five thousand entries against the handful a recipient or attachment
actually declares, so it bounds the read without being reachable by a real file. A constant
rather than configuration for the same reason ``MsgConfig`` has no field for it: nobody tunes the
size of a table of eight-byte integers."""

_CARRIED_HEADERS: Final = ("From", "To", "Cc", "Date", "Subject")
"""The headers copied from the transport block onto the reconstituted message.

The five :class:`~manicule.parsers.mail.MailParser` renders, and deliberately not the MIME
headers beside them. A ``Content-Type`` or ``Content-Transfer-Encoding`` from the original would
describe a body that no longer exists — this message's body is rebuilt from MAPI properties, and
a header asserting how it was once encoded would be false about the bytes it now labels.
"""

_ATTACHMENT_STORAGE = re.compile(r"^__attach_version1\.0_#[0-9A-Fa-f]{8}$")
_RECIPIENT_STORAGE = re.compile(r"^__recip_version1\.0_#[0-9A-Fa-f]{8}$")


class MsgParser:
    """Reads an Outlook ``.msg`` by converting it to a message and delegating."""

    media_types = MSG_MEDIA_TYPES
    profile = ParserProfile(name="msg", max_unlocated_ratio=0.05, max_pagelevel_ratio=None)

    def __init__(self, config: MsgConfig) -> None:
        self._config = config
        self._mail = MailParser(config.mail)

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Whatever the ``.eml`` parser makes of the reconstituted message."""
        async for block in self._mail.parse(self._converted(raw)):
            yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Re-convert and resolve through the same parser that produced the anchor.

        Deterministic rather than stored: the same file through the same reader gives the same
        message back, which is what makes a line anchor into a ``.msg`` exact rather than
        approximately right. It is the same argument §10 makes for pinning the HTML-to-text
        conversion, applied one layer out.
        """
        return await self._mail.resolve(anchor, self._converted(raw))

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        """The attachments, as the ``.eml`` parser sees them.

        They are reconstituted as MIME parts rather than read out of the MAPI storages here, so
        a ``.msg`` attachment and a ``.eml`` attachment become members through one code path —
        with one set of refusals, one depth budget and one identity scheme.
        """
        async for member in self._mail.expand(self._converted(raw)):
            yield member

    def _converted(self, raw: RawDocument) -> RawDocument:
        return RawDocument(
            source_id=raw.source_id,
            uri=raw.uri,
            media_type="message/rfc822",
            content=_as_rfc5322(raw, config=self._config),
            metadata=dict(raw.metadata),
        )


def _as_rfc5322(raw: RawDocument, *, config: MsgConfig) -> bytes:
    """One ``.msg`` as the message it describes."""
    data = raw.as_bytes()
    if not data.startswith(_OLE_SIGNATURE):
        msg = "not a compound file: an Outlook .msg begins with the OLE signature"
        raise ParseError(msg)
    import olefile  # noqa: PLC0415 - a parsing extra, kept out of the import boundary

    try:
        with olefile.OleFileIO(BytesIO(data)) as ole:
            return _message(ole, config=config)
    except ParseError:
        raise
    except Exception as exc:
        msg = f"the compound file could not be read ({type(exc).__name__}: {exc})"
        raise ParseError(msg) from exc


def _message(ole: object, *, config: MsgConfig) -> bytes:
    built = EmailMessage()
    headers = _text(ole, _TRANSPORT_HEADERS, limit=config.max_property_bytes)
    subject = _text(ole, _SUBJECT, limit=config.max_property_bytes)
    if headers:
        _carry(built, headers)
    else:
        # Drafts and Sent Items never traversed a transport, so there is no header block to
        # carry and the addresses have to be assembled from the properties that do exist. This
        # is the larger half of a real export rather than an edge case.
        _synthesize(ole, built, limit=config.max_property_bytes)
    if subject and "Subject" not in built:
        built["Subject"] = subject

    body = _text(ole, _BODY, limit=config.max_body_bytes)
    html = _binary(ole, _HTML_BODY, limit=config.max_body_bytes)
    if body:
        built.set_content(body)
    elif html is not None:
        built.set_content(html.decode("utf-8", errors="replace"), subtype="html")
    else:
        # A message with no body is a real shape — a calendar decline, a bounce stub — and the
        # `.eml` parser already emits the header block alone for it.
        built.set_content("")

    for name, payload, (maintype, subtype) in _attachments(ole, config=config):
        built.add_attachment(payload, maintype=maintype, subtype=subtype, filename=name)
    return built.as_bytes(policy=SMTP)


def _carry(built: EmailMessage, headers: str) -> None:
    """Copy the address and subject headers from the transport block, and nothing else."""
    original = message_from_string(headers)
    for name in _CARRIED_HEADERS:
        value = original.get(name)
        if value:
            built[name] = value


def _synthesize(ole: object, built: EmailMessage, *, limit: int) -> None:
    """Assemble From and To from the sender properties and the recipient table.

    The subject is the caller's to set, because it is set the same way whether or not there was
    a transport block to read it from.
    """
    sender = _address(
        _text(ole, _SENDER_SMTP, limit=limit) or _text(ole, _SENDER_EMAIL, limit=limit),
        _text(ole, _SENDER_NAME, limit=limit),
    )
    if sender:
        built["From"] = sender
    recipients: dict[str, list[str]] = {"To": [], "Cc": []}
    for entry, kind in _recipients(ole, limit=limit):
        recipients.setdefault(kind, []).append(entry)
    for name in ("To", "Cc"):
        if recipients.get(name):
            built[name] = ", ".join(recipients[name])


def _recipients(ole: object, *, limit: int) -> Iterator[tuple[str, str]]:
    for storage in _storages(ole, _RECIPIENT_STORAGE):
        address = _text(ole, _RECIPIENT_SMTP, prefix=storage, limit=limit) or _text(
            ole, _RECIPIENT_EMAIL, prefix=storage, limit=limit
        )
        entry = _address(address, _text(ole, _RECIPIENT_NAME, prefix=storage, limit=limit))
        if not entry:
            continue
        kind = {1: "To", 2: "Cc", 3: "Bcc"}.get(_long(ole, _RECIPIENT_TYPE, prefix=storage), "To")
        # Bcc is dropped rather than rendered: a header this reader invented, naming people the
        # message deliberately did not name to its recipients, is not a fact about the message.
        if kind != "Bcc":
            yield entry, kind


def _media_type(declared: str) -> tuple[str, str]:
    """``PidTagAttachMimeTag`` split for :meth:`EmailMessage.add_attachment`, or a safe default.

    Worth reading rather than labeling everything ``application/octet-stream``: the mail parser
    infers a member's type from its filename only when the part declares none, so an
    extensionless attachment — or one whose extension disagrees with what the sender said it was
    — otherwise reaches the parser chain as generic bytes and is routed nowhere.

    Anything that is not one plain ``type/subtype`` falls back to the default rather than being
    passed through, because the value is attacker-controlled and ``add_attachment`` puts it into
    a header verbatim.
    """
    maintype, _, subtype = declared.strip().lower().partition("/")
    if not (maintype.isascii() and maintype.isalnum() and subtype.isascii() and subtype):
        return "application", "octet-stream"
    if not subtype.replace("-", "").replace("+", "").replace(".", "").isalnum():
        return "application", "octet-stream"
    return maintype, subtype


def _attachments(ole: object, *, config: MsgConfig) -> Iterator[tuple[str, bytes, tuple[str, str]]]:
    storages = _storages(ole, _ATTACHMENT_STORAGE)
    if len(storages) > config.max_attachments:
        msg = (
            f"the message declares {len(storages)} attachments, above the "
            f"{config.max_attachments} ceiling. Raise parsers.msg.max_attachments to read it, "
            f"or leave it refused."
        )
        raise ParseError(msg)
    for ordinal, storage in enumerate(storages):
        payload = _binary(ole, _ATTACH_DATA, prefix=storage, limit=config.max_attachment_bytes)
        if payload is None:
            continue
        name = (
            _text(ole, _ATTACH_LONG_NAME, prefix=storage, limit=config.max_property_bytes)
            or _text(ole, _ATTACH_NAME, prefix=storage, limit=config.max_property_bytes)
            or f"attachment-{ordinal + 1}"
        )
        declared = _text(ole, _ATTACH_MIME_TAG, prefix=storage, limit=config.max_property_bytes)
        yield name, payload, _media_type(declared)


def _storages(ole: object, pattern: re.Pattern[str]) -> list[str]:
    """Matching top-level storages, in the order their eight-digit suffix gives them.

    Sorted rather than taken as listed, because the suffix is the ordinal and the directory is
    not obliged to be in it — and two expansions of one file that disagreed about which
    attachment came first would give the same message two different member sets.
    """
    entries = {
        entry[0]
        for entry in _listdir(ole)
        if len(entry) > 1 and pattern.match(entry[0]) is not None
    }
    return sorted(entries)


def _listdir(ole: object) -> Sequence[Sequence[str]]:
    listdir = ole.listdir  # pyright: ignore[reportAttributeAccessIssue] - olefile has no stubs
    return listdir(streams=True, storages=False)


def _stream(ole: object, tag: str, kind: str, *, prefix: str | None, limit: int) -> bytes | None:
    """One property stream, read to at most ``limit + 1`` bytes.

    The extra byte is the whole point: enough to know the property is over its ceiling, and not
    enough to pay for it. Reading the stream whole and measuring afterwards spends exactly the
    memory the ceiling exists to protect, which would make the limits in
    :class:`~manicule.parsers.config.MsgConfig` a report on the allocation rather than a bound on
    it — and a ``.msg`` is a file from the corpus, so the size it declares is a number somebody
    else wrote.
    """
    name = f"__substg1.0_{tag}{kind}"
    path = name if prefix is None else f"{prefix}/{name}"
    exists = ole.exists  # pyright: ignore[reportAttributeAccessIssue] - olefile has no stubs
    if not exists(path):
        return None
    with ole.openstream(path) as handle:  # pyright: ignore[reportAttributeAccessIssue]
        return handle.read(limit + 1)


def _text(ole: object, tag: str, *, prefix: str | None = None, limit: int) -> str:
    """A string property in whichever of its two spellings the file used."""
    unicode_bytes = _stream(ole, tag, _UNICODE, prefix=prefix, limit=limit)
    if unicode_bytes is not None:
        return _bounded(unicode_bytes, limit, tag).decode("utf-16-le", errors="replace")
    ansi_bytes = _stream(ole, tag, _ANSI, prefix=prefix, limit=limit)
    if ansi_bytes is not None:
        return _bounded(ansi_bytes, limit, tag).decode("cp1252", errors="replace")
    return ""


def _binary(ole: object, tag: str, *, prefix: str | None = None, limit: int) -> bytes | None:
    """A ``PT_BINARY`` property, which is the one that catches people.

    ``PidTagHtml`` is binary, so the stream is ``…10130102``; a reader that assumed
    ``…1013001F`` by analogy with the plain body finds nothing and concludes the message has no
    HTML part (``docs/parsing.md`` §10).
    """
    found = _stream(ole, tag, _BINARY, prefix=prefix, limit=limit)
    return None if found is None else _bounded(found, limit, tag)


def _long(ole: object, tag: int, *, prefix: str) -> int:
    """A ``PT_LONG`` property from a storage's properties stream, or ``0``."""
    path = f"{prefix}/{_PROPERTIES}"
    exists = ole.exists  # pyright: ignore[reportAttributeAccessIssue] - olefile has no stubs
    if not exists(path):
        return 0
    with ole.openstream(path) as handle:  # pyright: ignore[reportAttributeAccessIssue]
        data: bytes = handle.read(_MAX_PROPERTIES_BYTES)
    for at in range(_PROPERTIES_HEADER, len(data) - _PROPERTY_ENTRY + 1, _PROPERTY_ENTRY):
        if int.from_bytes(data[at : at + 4], "little") == tag:
            return int.from_bytes(data[at + 8 : at + 12], "little")
    return 0


def _bounded(data: bytes, limit: int, tag: str) -> bytes:
    if len(data) > limit:
        msg = (
            f"MAPI property {tag} is {len(data)} bytes, above the {limit}-byte ceiling. "
            f"Raise the matching parsers.msg limit to read it, or leave it refused."
        )
        raise ParseError(msg)
    return data


def _address(email: str, display: str) -> str:
    """One recipient as a header value, from whichever halves the file supplied."""
    email, display = email.strip(), display.strip()
    if email and display:
        return f'"{display}" <{email}>'
    return email or display


__all__ = ["MsgParser"]
