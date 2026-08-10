"""``.eml`` messages, anchored to the lines of one canonically-composed text.

A multipart message has several candidate bodies, so one rule makes the parser deterministic
(``docs/parsing.md`` §10):

    The canonical body is the **first ``text/plain`` part in depth-first order**; failing that,
    the **first ``text/html`` part**, run through the HTML parser.

**One coordinate space, composed here.** The specification says line numbers address the
decoded body part, and says separately that the headers become a block "with its own line
span". Those are two different coordinate spaces, and a :class:`~manicule.core.anchors.LineAnchor`
carries no way to say which one it is in — ``lines 1-4`` would mean the headers to one reader
and the opening of the body to another, and both would resolve to something plausible. So this
parser defines the message's canonical text as **the rendered header block, one blank line,
then the canonical body**, and every anchor addresses that. :meth:`MailParser.resolve` rebuilds
it from the same bytes by the same rules, which is what makes the anchor exact rather than
approximately right.

**The HTML-to-text conversion is pinned**, and :data:`HTML_TO_TEXT_VERSION` is its identity.
For an HTML-only body the line numbers address the *converted* text rather than the source
bytes, so a converter upgrade shifts every anchor in every HTML email — round-tripping today,
pointing at the wrong paragraph after a dependency bump, with no test failing in between. The
version therefore belongs in
:attr:`ChunkFingerprint.version <manicule.core.fingerprints.ChunkFingerprint.version>`, by way
of ``StructuralChunker(version_components=...)``, where a change to it refuses to run against
a corpus built with the old one.

**Quoted reply chains are kept.** Trimming them is a retrieval optimisation with a real
downside — the quoted text is frequently the only statement of the thing being replied to —
and it is not a parsing decision.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import (
    BlockKind,
    DocumentStatus,
    Metadata,
    ParsedBlock,
    RawDocument,
)
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.core.protocols import read_blocks
from manicule.parsers.base import ParserProfile, lines_of, resolve_lines
from manicule.parsers.config import MAIL_MEDIA_TYPES, MailConfig, html_text_version
from manicule.parsers.expansion import (
    CONTAINER_DEPTH,
    OCTET_STREAM,
    PATH_HASHES,
    TREE_BYTES,
    TREE_MEMBERS,
    ExpandedMember,
    MemberFailure,
    MemberOutcome,
    container_depth_of,
    inner_path,
    media_type_for,
    member_source_id,
    member_uri,
    path_hashes_of,
    tree_bytes_of,
    tree_members_of,
)
from manicule.parsers.plaintext import paragraph_spans
from manicule.parsers.web import WebConfig, WebParser

__all__ = [
    "MAIL_MEDIA_TYPES",
    "MAIL_SCHEME",
    "MailConfig",
    "MailParser",
]

MAIL_SCHEME = "mail"
"""The scheme a member address carries: ``mail:<message-uri>!/report.pdf``."""

HEADER_FIELDS = ("From", "To", "Cc", "Date", "Subject")
"""The headers the canonical text renders, in this order, present ones only.

A fixed order rather than the source's, because the source's order is a property of whichever
mail transfer agent touched the message last, and every anchor in the message is measured from
the height of this block.
"""

_FOLD = re.compile(r"\r\n[ \t]+|\r[ \t]+|\n[ \t]+|[\r\n]+")
"""Line breaks inside a header value. A header is logically one line however it was folded for
transport, and a value spanning two lines would put the anchors of the whole message half a
line out."""


@dataclass(frozen=True, slots=True)
class _Body:
    """The canonical body: its text, and which part it came from."""

    text: str
    part_path: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _Canonical:
    """One reading of a message: the text every anchor addresses, and what is in it."""

    lines: tuple[str, ...]
    header_height: int
    subject: str
    body: _Body | None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class MailParser:
    """Parses an ``.eml`` message into a header block and the body's paragraphs."""

    media_types = MAIL_MEDIA_TYPES
    profile = ParserProfile(name="email", max_unlocated_ratio=0.05, max_pagelevel_ratio=None)
    """A small unlocated budget the parser does not currently spend: every block it emits
    carries a line span into the canonical text. It is declared because the part-selection
    rule can be defeated — a message whose only body is an attachment-dispositioned part has
    no canonical body — and a budget of zero would turn that into a suite failure rather than
    a measured degradation."""

    def __init__(self, config: MailConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield the header block, then one block per body paragraph.

        Raises:
            ParseError: The bytes are not an RFC 5322 message, or its body part does not
                decode with the character set it declares.
        """
        canonical = await self._read(raw)
        heading_path = (canonical.subject,) if canonical.subject else ()
        if canonical.header_height:
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text="\n".join(canonical.lines[: canonical.header_height]),
                anchor=LineAnchor(start=1, end=canonical.header_height),
                heading_path=heading_path,
                # The subject is the only structure an email has, so it is the document's
                # title as well as the first element of the heading path (§10).
                metadata=self._header_metadata(canonical),
            )
        lines = list(canonical.lines)
        for start, end in paragraph_spans(lines, max_lines=self._config.max_block_lines):
            if end <= canonical.header_height:
                continue
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text="\n".join(lines[start - 1 : end]),
                anchor=LineAnchor(start=start, end=end),
                heading_path=heading_path,
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the canonical text ``anchor`` addresses, or ``None`` if it addresses none.

        Recomposes the canonical text from ``raw`` — including running an HTML-only body back
        through the same pinned conversion — rather than consulting anything :meth:`parse`
        left behind. Resolving against the parser's own memory of a document verifies nothing
        about the document.
        """
        if not isinstance(anchor, LineAnchor):
            return None
        return resolve_lines((await self._read(raw)).text, anchor)

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        """Yield each attachment as a document of its own.

        A PDF is a PDF wherever it arrived from, so an attachment goes back through the whole
        parser chain rather than through anything mail-specific (§10).

        Raises:
            ParseError: The bytes are not an RFC 5322 message.
        """
        message = _message_of(raw)
        canonical = await self._read(raw)
        body_path = canonical.body.part_path if canonical.body else ""
        depth = container_depth_of(raw) + 1
        hashes = path_hashes_of(raw)
        members = tree_members_of(raw)
        used: set[str] = set()

        for path, part in _walk(message):
            if path == body_path or not _is_attachment(part):
                continue
            name = _attachment_path(part, path, used)
            used.add(name)
            members += 1
            if members > self._config.max_members:
                # A message is a container, and every other container here counts its members.
                # Without this a message with a hundred thousand parts becomes a hundred
                # thousand documents, and an archive → message → archive chain has no member
                # ceiling on the middle hop.
                yield self._refused(
                    raw,
                    name,
                    depth,
                    f"attachment count exceeded: {members} across this container tree, and "
                    f"the limit is {self._config.max_members}. Raise maxMembers to index the "
                    f"rest.",
                    DocumentStatus.FAILED,
                )
                return
            if not self._config.expand_attachments:
                yield self._refused(
                    raw,
                    name,
                    depth,
                    "attachment expansion is disabled by configuration — set "
                    "expandAttachments to index attachments as documents of their own",
                    DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                continue
            if depth > self._config.max_depth:
                yield self._refused(
                    raw,
                    name,
                    depth,
                    f"archive nesting depth exceeded — this attachment is {depth} containers "
                    f"deep and the limit is {self._config.max_depth}",
                    DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                continue
            payload = _payload_of(part)
            digest = content_hash(payload)
            if digest in hashes:
                yield self._refused(
                    raw,
                    name,
                    depth,
                    "container cycle detected — this attachment is byte-identical to one of "
                    "the documents containing it",
                    DocumentStatus.FAILED,
                )
                continue
            yield ExpandedMember(
                source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
                uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
                raw=RawDocument(
                    source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
                    uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
                    media_type=_member_media_type(part, name),
                    content=payload,
                    metadata=self._member_metadata(raw, depth, digest, members, part),
                ),
                depth=depth,
                metadata={"member_filename": name, "member_part": path},
            )

    # --- reading -------------------------------------------------------------------------

    async def _read(self, raw: RawDocument) -> _Canonical:
        message = _message_of(raw)
        headers = [
            f"{field}: {_unfolded(str(value))}"
            for field in HEADER_FIELDS
            if (value := message.get(field)) is not None and str(value).strip()
        ]
        subject = _unfolded(str(message.get("Subject", ""))).strip()
        body = await self._body_of(message, raw.uri)
        lines = list(headers)
        if body is not None and body.text:
            if lines:
                lines.append("")
            lines.extend(_normalised_lines(body.text))
        return _Canonical(
            lines=tuple(lines), header_height=len(headers), subject=subject, body=body
        )

    async def _body_of(self, message: EmailMessage, uri: str) -> _Body | None:
        """The canonical body, by the depth-first rule that makes this parser deterministic."""
        html: tuple[str, EmailMessage] | None = None
        for path, part in _walk(message):
            if part.get_content_maintype() == "multipart" or _is_attachment(part):
                continue
            if part.get_content_type() == "text/plain":
                return _Body(text=_text_payload(part, uri), part_path=path, media_type="text/plain")
            if html is None and part.get_content_type() == "text/html":
                html = (path, part)
        if html is None:
            return None
        path, part = html
        return _Body(
            text=await _html_to_text(_text_payload(part, uri), uri),
            part_path=path,
            media_type="text/html",
        )

    def _header_metadata(self, canonical: _Canonical) -> Metadata:
        metadata: Metadata = {"title": canonical.subject} if canonical.subject else {}
        if canonical.body is not None:
            metadata["body_media_type"] = canonical.body.media_type
            if canonical.body.media_type == "text/html":
                metadata["html_to_text_version"] = html_text_version()
        return metadata

    def _member_metadata(
        self, raw: RawDocument, depth: int, digest: str, members: int, part: EmailMessage
    ) -> Metadata:
        """What a member needs to know about the tree it is in, and about itself.

        The byte budgets travel unchanged: an attachment is already resident in the message we
        were handed, so there is no streaming decision to make here, and inflating the counter
        for bytes nobody streamed would spend an archive's budget on an email.
        """
        return {
            CONTAINER_DEPTH: depth,
            PATH_HASHES: [*path_hashes_of(raw), digest],
            TREE_BYTES: tree_bytes_of(raw),
            TREE_MEMBERS: members,
            "member_content_type": part.get_content_type(),
        }

    def _refused(
        self,
        raw: RawDocument,
        name: str,
        depth: int,
        reason: str,
        status: DocumentStatus,
    ) -> MemberFailure:
        return MemberFailure(
            source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
            uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
            status=status,
            reason=reason,
            depth=depth,
            metadata={"member_filename": name},
        )


# --- the message -----------------------------------------------------------------------


def _message_of(raw: RawDocument) -> EmailMessage:
    """Parse the bytes as an RFC 5322 message, declining anything that is not one.

    Raises:
        ParseError: There are no header fields, so this is not a message. Declining rather
            than failing: the next parser in the chain may well be able to index it as text.
    """
    data = raw.as_bytes()
    if b"\x00" in data[:_HEADER_SAMPLE]:
        msg = (
            f"{raw.uri}: declining — the first {_HEADER_SAMPLE} bytes contain a NUL, and RFC "
            f"5322 header fields are text. This is not an .eml file."
        )
        raise ParseError(msg)
    message = BytesParser(policy=policy.default).parsebytes(data)
    if not message.keys():
        msg = (
            f"{raw.uri}: declining — no RFC 5322 header fields were found, so this is not a "
            f"message. Check the media type it was routed with."
        )
        raise ParseError(msg)
    return message


_HEADER_SAMPLE = 4096
"""How far in to look for a NUL before concluding this is not a message. Headers come first
and are text; a binary attachment further down is base64 by the time it reaches us."""


def _walk(message: EmailMessage) -> Iterator[tuple[str, EmailMessage]]:
    """Every part in depth-first order, each with the dotted path that addresses it.

    Written out rather than taken from :meth:`email.message.Message.walk` because the path is
    needed: two attachments may share a filename, and the part path is the only other address
    a message publishes for them.
    """
    yield from _descend(message, "1")


def _descend(part: EmailMessage, path: str) -> Iterator[tuple[str, EmailMessage]]:
    yield path, part
    if part.get_content_maintype() != "multipart":
        return
    payload = part.get_payload()
    if not isinstance(payload, list):
        return
    for index, child in enumerate(payload, start=1):
        if isinstance(child, EmailMessage):
            yield from _descend(child, f"{path}.{index}")


def _is_attachment(part: EmailMessage) -> bool:
    """Whether a part is an attachment rather than a candidate body.

    A ``text/plain`` part with a filename is a file someone attached, not the message they
    wrote, and treating it as the body would make the canonical text depend on what was
    attached to the message.

    **An enclosed message counts, whatever it is labelled.** A ``message/rfc822`` part
    frequently carries neither a disposition nor a filename — ``multipart/digest`` never does,
    and inline forwards from several mail clients do not either. Without this it satisfies no
    branch anywhere: :func:`_body_of` skips it because it is not ``text/*``, :func:`_walk`
    does not descend into it because it is not ``multipart``, and it is not an attachment — so
    a digest of twenty messages indexes as a header block and nothing else, reporting no
    failure. ``expansion.py`` is explicit that failure is an outcome and never an omission,
    and this is the one place that was quietly an omission.
    """
    return (
        part.get_content_disposition() == "attachment"
        or part.get_filename() is not None
        or part.get_content_maintype() == "message"
    )


def _payload_of(part: EmailMessage) -> bytes:
    """A part's bytes with its transfer encoding removed.

    An enclosed message is the exception: its payload is a parsed message rather than an
    encoded octet stream, so ``get_payload(decode=True)`` returns ``None`` for it. Serialising
    the enclosed message is what makes it a document the chain can parse — and it has to be
    the *enclosed* message rather than the part, or the member would carry the wrapper's
    headers as well as its own.
    """
    if part.get_content_maintype() == "message":
        enclosed = part.get_payload()
        if isinstance(enclosed, list) and enclosed:
            first = enclosed[0]
            if isinstance(first, EmailMessage):
                return first.as_bytes()
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else b""


def _text_payload(part: EmailMessage, uri: str) -> str:
    """A text part decoded with the character set it declares.

    Raises:
        ParseError: The declared character set is unknown, or the bytes are not in it.
            Guessing another encoding would put mojibake into the index under a citation that
            says it is a quotation.
    """
    charset = part.get_content_charset() or "utf-8"
    try:
        return _payload_of(part).decode(charset)
    except (LookupError, UnicodeDecodeError) as exc:
        msg = (
            f"{uri}: the {part.get_content_type()} body declares charset {charset!r} and does "
            f"not decode as it ({exc}). Fix the message's Content-Type, or route this media "
            f"type to the plaintext parser."
        )
        raise ParseError(msg) from exc


async def _html_to_text(html: str, uri: str) -> str:
    """The pinned conversion an HTML-only body's line numbers address.

    One rule, versioned by :data:`HTML_TO_TEXT_VERSION`: take the blocks the HTML parser
    yields and join them with a blank line. The HTML parser is imported rather than
    reimplemented, so an email body and an HTML page are read the same way and only have to be
    right once.
    """
    source = RawDocument(
        source_id=uri, uri=f"{uri}#html-body", media_type="text/html", content=html
    )
    # Drained through ``read_blocks`` rather than a comprehension over ``parse``: a stream
    # abandoned part-way stays suspended holding whatever it had open at the ``yield``, and
    # CPython finalises it late, from a loop that may already be closed.
    blocks = await read_blocks(WebParser(WebConfig()), source)
    return "\n\n".join(block.text for block in blocks)


def _unfolded(value: str) -> str:
    """A header value as one line, however it was folded for transport."""
    return _FOLD.sub(" ", value).strip()


def _normalised_lines(text: str) -> list[str]:
    """Body lines with transport line endings reduced to the one this module counts.

    Part of the pinned composition: a message uses CRLF, ``lines_of`` splits on ``\\n``, and a
    stray carriage return at the end of every line would appear in every quotation.
    """
    return lines_of(text.replace("\r\n", "\n").replace("\r", "\n"))


def _attachment_path(part: EmailMessage, path: str, used: set[str]) -> str:
    """The inner path that identifies an attachment inside its message.

    The filename, because that is what a person would cite and what stays the same when a
    message is re-fetched. A message with two attachments of the same name publishes no other
    address for them, so the second one is qualified by its MIME part path — which is
    positional, and is used only where the alternative is two documents with one identity.
    """
    filename = part.get_filename()
    candidate = inner_path(filename) if filename else None
    if candidate is None:
        return f"part-{path}"
    return candidate if candidate not in used else f"{path}/{candidate}"


def _member_media_type(part: EmailMessage, name: str) -> str:
    """What the part declares, falling back to what its name implies.

    The declared type wins because ``docs/parsing.md`` §6.1 puts a type the source states above
    a filename extension — except for ``application/octet-stream``, which is not a claim but
    the absence of one, and is the one type sniffing and naming are allowed to replace.
    """
    declared = part.get_content_type()
    if declared and declared != OCTET_STREAM:
        return declared
    return media_type_for(name)
