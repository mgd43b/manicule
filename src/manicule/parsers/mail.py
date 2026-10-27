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
converter is therefore part of this parser's lineage — ``selectolax`` among its libraries in
:data:`~manicule.parsers.versions.PARSERS`, and the join rule among its ``rules`` — so a change
to either re-parses the mail it affects. It used to be a suffix on the corpus-wide
:attr:`ChunkFingerprint.version <manicule.core.fingerprints.ChunkFingerprint.version>`, where a
``selectolax`` release refused every ingest into every index, mail or not. The version is still
stamped on each block as ``html_to_text_version``, for anyone reading one.

**Quoted reply chains are kept.** Trimming them is a retrieval optimization with a real
downside — the quoted text is frequently the only statement of the thing being replied to —
and it is not a parsing decision.
"""

from __future__ import annotations

import base64
import quopri
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from email import policy
from email.errors import MissingHeaderBodySeparatorDefect
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
    PATH_HASHES,
    TREE_BYTES,
    TREE_MEMBERS,
    ExpandedMember,
    MemberFailure,
    MemberOutcome,
    container_depth_of,
    inner_path,
    member_source_id,
    member_uri,
    path_hashes_of,
    tree_bytes_of,
    tree_members_of,
)
from manicule.parsers.mediatype import (
    ALIASES,
    GENERIC,
    UNINDEXED_MAIN_TYPES,
    resolve_member_media_type,
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
                    truncates=True,
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
            declared = ALIASES.get(part.get_content_type(), part.get_content_type())
            if declared.split("/", 1)[0] in UNINDEXED_MAIN_TYPES:
                yield self._unindexed(raw, name, depth, declared)
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
            media_type = resolve_member_media_type(declared, name, payload)
            transcoded = False
            if media_type == declared and declared in GENERIC:
                # Nothing established what the bytes are, which for a text attachment in a
                # multi-byte character set is the NULs of UTF-16 reading as binary. Resolve again
                # over the declared character set's reading, and keep it only if it is text.
                decoded = _transcoded(part, payload)
                if decoded != payload:
                    retried = resolve_member_media_type(declared, name, decoded)
                    if retried.startswith("text/"):
                        media_type, payload, transcoded = retried, decoded, True
            if media_type.split("/", 1)[0] in UNINDEXED_MAIN_TYPES:
                yield self._unindexed(raw, name, depth, media_type)
                continue
            if not transcoded:
                payload = _as_utf8(part, media_type, payload)
            yield ExpandedMember(
                source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
                uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
                raw=RawDocument(
                    source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
                    uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
                    media_type=media_type,
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
            lines.extend(_normalized_lines(body.text))
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
        *,
        truncates: bool = False,
    ) -> MemberFailure:
        return MemberFailure(
            source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
            uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
            status=status,
            reason=reason,
            depth=depth,
            truncates=truncates,
            metadata={"member_filename": name},
        )

    def _unindexed(self, raw: RawDocument, name: str, depth: int, media_type: str) -> MemberFailure:
        """An attachment of a kind no parser reads, refused before its bytes are decoded.

        The row stays, so the member set of the message is still complete, and the bytes do not:
        a refused member is stored without retaining them, where an unsupported one that went
        through the chain would have been retained and then declined (``docs/parsing.md`` §10.4).
        """
        return MemberFailure(
            source_id=member_source_id(raw.source_id, name, scheme=MAIL_SCHEME),
            uri=member_uri(raw.uri, name, scheme=MAIL_SCHEME),
            status=DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            reason=(
                f"{media_type} attachments are not indexed — no text is extracted from images, "
                f"video or audio, so the bytes were not kept"
            ),
            depth=depth,
            metadata={"member_filename": name, "member_content_type": media_type},
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


_NAMELESS_ATTACHMENT_SUBTYPES = frozenset({"calendar", "markdown", "csv", "rfc822-headers", "xml"})
"""Text subtypes that are a document even without a filename. Deliberately a list rather than
"any text part that is not plain or HTML": mailers add machine alternatives (``text/x-amp-html``)
that are neither the message nor anything worth a row in every mailbox message."""


def _is_attachment(part: EmailMessage) -> bool:
    """Whether a part is an attachment rather than a candidate body.

    A ``text/plain`` part with a filename is a file someone attached, not the message they
    wrote, and treating it as the body would make the canonical text depend on what was
    attached to the message.

    **An enclosed message counts, whatever it is labeled.** A ``message/rfc822`` part
    frequently carries neither a disposition nor a filename — ``multipart/digest`` never does,
    and inline forwards from several mail clients do not either. Without this it satisfies no
    branch anywhere: :func:`_body_of` skips it because it is not ``text/*``, :func:`_walk`
    does not descend into it because it is not ``multipart``, and it is not an attachment — so
    a digest of twenty messages indexes as a header block and nothing else, reporting no
    failure. ``expansion.py`` is explicit that failure is an outcome and never an omission,
    and this is the one place that was quietly an omission.

    **So does a nameless text part of a kind that is a document.** A calendar invitation is a
    ``text/calendar`` alternative beside the ``text/plain`` one, with no filename and no
    disposition. It is never the canonical body (only ``text/plain`` and ``text/html`` are), so
    treating it as anything but an attachment made it, like the enclosed message, vanish.
    """
    return (
        part.get_content_disposition() == "attachment"
        or part.get_filename() is not None
        or part.get_content_maintype() == "message"
        or (
            part.get_content_maintype() == "text"
            and part.get_content_subtype() in _NAMELESS_ATTACHMENT_SUBTYPES
        )
    )


def _payload_of(part: EmailMessage) -> bytes:
    """A part's bytes with its transfer encoding removed.

    An enclosed message is the exception: its payload is a parsed message rather than an
    encoded octet stream, so ``get_payload(decode=True)`` returns ``None`` for it. Serializing
    the enclosed message is what makes it a document the chain can parse — and it has to be
    the *enclosed* message rather than the part, or the member would carry the wrapper's
    headers as well as its own.

    **Two shapes of enclosed part are not one message.** A ``message/delivery-status`` is a list
    of header blocks (one for the reporting server, one per recipient), and reading only the
    first dropped every recipient's status. And a ``message/rfc822`` declared
    ``Content-Transfer-Encoding: base64`` — forbidden by RFC 2046 and sent by real clients
    regardless — is parsed as though its base64 text were headers, so the transfer encoding has
    to be undone on the serialized text before it is a message.
    """
    if part.get_content_maintype() == "message":
        enclosed = part.get_payload()
        if isinstance(enclosed, list) and enclosed:
            encoded = (
                str(part.get("Content-Transfer-Encoding", "")).strip().lower() in _ENCODED_TRANSFERS
            )
            blocks = [
                _unparsed(block) if encoded else block.as_bytes()
                for block in enclosed
                if isinstance(block, EmailMessage)
            ]
            if not blocks:
                return b""
            return _undone(part, b"\n".join(blocks) if len(blocks) > 1 else blocks[0])
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else b""


_ENCODED_TRANSFERS = frozenset({"base64", "quoted-printable"})


def _unparsed(block: EmailMessage) -> bytes:
    """The text the parser read as a message but which is a transfer-encoded body.

    Serializing the parsed message inserts the blank line between header and body that the source
    never had, which is harmless to base64 and corrupts quoted-printable: a soft-wrapped line
    (``Subject: =?...=`` then ``rest``) is read as the end of the headers, and a separator put
    in front of the continuation splits the line. The parser records exactly that case as a
    missing separator, so the original text is the raw header lines followed directly by the body.
    """
    payload = block.get_payload()
    body = payload if isinstance(payload, str) else ""
    headers = "".join(f"{name}: {value}\n" for name, value in block.raw_items())
    separated = not any(
        isinstance(defect, MissingHeaderBodySeparatorDefect) for defect in block.defects
    )
    return (headers + ("\n" if separated and headers else "") + body).encode(
        "utf-8", "surrogateescape"
    )


def _undone(part: EmailMessage, serialized: bytes) -> bytes:
    """``serialized`` with the part's declared base64 or quoted-printable encoding removed."""
    encoding = str(part.get("Content-Transfer-Encoding", "")).strip().lower()
    try:
        if encoding == "base64":
            return base64.b64decode(serialized)
        if encoding == "quoted-printable":
            return quopri.decodestring(serialized)
    except ValueError:
        return serialized
    return serialized


def _as_utf8(part: EmailMessage, media_type: str, payload: bytes) -> bytes:
    """A text attachment's bytes in UTF-8, using the character set its part declares.

    The member is a document of its own and is parsed without the part that carried it, so the
    declared character set would otherwise be lost — and a Latin-1 or UTF-16 text file, which
    the plaintext parser declines as binary, would be reported unsupported.
    """
    return _transcoded(part, payload) if media_type.startswith("text/") else payload


def _transcoded(part: EmailMessage, payload: bytes) -> bytes:
    """``payload`` read in the part's declared character set and written as UTF-8.

    A declared set that does not decode the bytes (a mislabeled ``us-ascii`` that is really
    UTF-8, most often) leaves them as they were, because the label is the sender's claim and the
    bytes are the evidence.
    """
    charset = part.get_content_charset()
    if not charset or charset.lower().replace("_", "-") in {"utf-8", "utf8", "us-ascii", "ascii"}:
        return payload
    try:
        return payload.decode(charset).encode("utf-8")
    except (LookupError, UnicodeError):
        return payload


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
    # CPython finalizes it late, from a loop that may already be closed.
    blocks = await read_blocks(WebParser(WebConfig()), source)
    return "\n\n".join(block.text for block in blocks)


def _unfolded(value: str) -> str:
    """A header value as one line, however it was folded for transport."""
    return _FOLD.sub(" ", value).strip()


def _normalized_lines(text: str) -> list[str]:
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
