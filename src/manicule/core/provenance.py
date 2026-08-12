"""Authoritative source metadata: what a document *is*, apart from where this machine keeps it.

A locally mirrored page is two things at once, and a citation that conflates them is useless in
both directions. ``123456.html`` under some corpus root is a **snapshot**: it has a path
somewhere, and a moment it was taken. The thing it is a snapshot *of* has a title somebody
wrote, a canonical address a reader can open, an identifier its publisher assigns and a version
that moves when the page is edited. Citing the first as though it were the second produces a
reference that is perfectly accurate about a file nobody else has, and says nothing whatever
about the document — which is the defect this module exists to remove.

**Two models, not one with more fields, and the split is the design.**

:class:`SourceMetadata` has nowhere to put a local path.
    Every field on it is a claim about the publication. A connector that only knows where a
    file sits on this disk cannot fill any of them in, so it cannot accidentally assert that
    the file is the publication.

:class:`LocalSnapshot` has nowhere to put a canonical URI.
    Every field on it is about this machine's copy. Nothing that reads it can mistake it for
    the document's own identity.

That is the same reasoning
:class:`~manicule.app.results.SharedCitationLabel` uses against
:class:`~manicule.app.results.AnswerCitation` — a *different shape*, rather than the same shape
with some fields blanked, because a field that does not exist cannot be populated by a caller
who forgot which one they were holding. Both identities are preserved and neither is
representable as the other.

**Three timestamps, three names, never folded together.**
:attr:`SourceMetadata.modified_at` is when the *document* was last edited, at its source.
:attr:`LocalSnapshot.retrieved_at` is when this copy was taken. ``documents.indexed_at`` is
when manicule indexed it. They routinely differ by months, and the failure mode of collapsing
any two is a document that looks freshly revised because somebody re-ran an import.

**Nothing here is specific to one documentation product.** That is a requirement rather than a
preference: a wiki connector, an exported-site connector and a plain sidecar manifest all have
these facts under their own names — page ID, space key, ancestors — and each of those is one of
these fields spelled locally. A field named after any one product would either go unused by the
others or drag its vocabulary into the core, and the core is what every surface renders.

**Where the record lives.** In ``documents.metadata``, under :data:`PROVENANCE_KEY`, read back
through :attr:`manicule.core.content.Document.provenance`. Not a column: ``docs/storage.md``
§6.4 earns ``parse_fp`` its column on the grounds that *change detection reads it* and so it
must be queryable per document, and nothing queries this — it is read at citation time from a
row that has already been loaded. The reserved-key-plus-property shape is
:attr:`manicule.core.content.Chunk.lang`'s, for its reason too: the accessor and the stored
value cannot disagree when there is only one of them.

**It is validated on every read, not only on write.** The record originates in a file inside
the corpus, which makes it attacker-controlled input that has been through a database since
anyone last looked at it. Re-validating costs one model construction per document and is what
stops a ``javascript:`` URI reaching a rendered citation on the strength of having once been
stored.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

PROVENANCE_KEY: Final = "source_provenance"
"""The ``documents.metadata`` key the record is stored under.

One name, exported, because two spellings of it are two records and the citation would read
whichever one it happened to look for.
"""

CITABLE_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
"""URI schemes a canonical address may use.

**An allowlist, and the whole security property of this module rests on its being one.** The
canonical URI is rendered as a link on the browser surface and printed by the command line, so
the denylist formulation — "not ``javascript:``" — has to enumerate every scheme a browser will
execute, and gets ``data:``, ``vbscript:`` and whatever the next one is wrong by omission. It
also has to exclude ``file:``, which is not executable and *is* the exact confusion this module
exists to prevent: a manifest naming a local path as the document's canonical address.

Two schemes are enough. A canonical address is where a reader goes to see the published
document, and that is a web address or it is not a canonical address.
"""

MAX_FIELD_CHARS: Final = 1024
"""Longest a declared string may be.

Not arbitrary caution: these strings are rendered into a citation on every surface, and a title
of a few megabytes is a page nobody can read and a terminal nobody can scroll. Refused rather
than truncated, because a truncated title is a *different* title presented as the document's
own.
"""

MAX_SECTION_DEPTH: Final = 32
"""Longest a declared hierarchy may be.

The breadcrumb elides anything past its token budget anyway
(:func:`manicule.chunking.breadcrumb.render`), so a thousand-element path buys nothing and
costs a row that is mostly one field.
"""

_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$")
"""``type/subtype``, and nothing else.

Deliberately not a full RFC 6838 parse with parameters. What this field is for is routing and
display; a declared value with a ``;charset=`` on it is a value somebody expected to be
interpreted, and interpreting half of it is worse than refusing all of it.
"""


def _printable(value: str, *, field: str) -> str:
    """``value`` with its surrounding space removed, refused if it holds a control character.

    Control characters in a declared string are not a hypothetical. ``\\x1b`` is the opening
    byte of an ANSI escape sequence, and every one of these fields is printed to a terminal by
    ``manicule search`` — so a title is a route to repositioning the cursor, recolouring the
    output or clearing the screen of a person reading a citation. The browser surface escapes
    HTML and would not have caught it, because it is not markup.

    ``NUL`` is refused by the same check and for a different reason: it terminates a C string,
    and it travels through JSON, SQLite and an FTS5 index that all handle it differently.

    Raises:
        ValueError: Naming the field and the offending code point, because "invalid title" does
            not tell whoever wrote the manifest which character to remove.
    """
    if len(value) > MAX_FIELD_CHARS:
        msg = f"{field} is {len(value)} characters, over the {MAX_FIELD_CHARS}-character limit"
        raise ValueError(msg)
    for character in value:
        if unicodedata.category(character) == "Cc":
            msg = (
                f"{field} contains the control character U+{ord(character):04X}, which cannot "
                f"appear in a citation. Remove it from the manifest."
            )
            raise ValueError(msg)
    return value.strip()


def _aware(value: datetime | None, *, field: str) -> datetime | None:
    """``value``, refused if it is naive.

    A naive timestamp is not a moment: read as UTC it is wrong by the mirror host's offset, and
    read as local time it is wrong on every other machine. :class:`.sources.Watermark` refuses
    one for the same reason, and a source modification time that is silently wrong by a day is
    exactly the sort of thing nobody notices until it is used to decide which of two versions
    is newer.

    Raises:
        ValueError: When ``value`` carries no offset.
    """
    if value is not None and value.tzinfo is None:
        msg = f"{field} must carry a UTC offset; {value.isoformat()} does not say which zone"
        raise ValueError(msg)
    return value


class SourceMetadata(BaseModel):
    """What the publication says about itself.

    Every field is a claim about the document at its source, and there is deliberately nowhere
    here to record a local path, so no connector can assert that a file on this disk is the
    published thing.

    All fields are optional and that is intentional: a mirroring tool that knows a page's title
    and URL but not its version should be able to say so, and a required field it had to invent
    would be a fabricated one. What is *not* permitted is a record that claims nothing at all —
    see :meth:`_says_something`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(
        default="",
        description="The document's own title, as its publisher wrote it. Preferred over a "
        "local filename in every user-facing citation, which is the whole point: "
        "``123456.html`` is a fact about a mirror and not a description of anything.",
    )
    canonical_uri: str = Field(
        default="",
        description="Where a reader goes to see the published document. Restricted to "
        f"{sorted(CITABLE_SCHEMES)} — a ``file:`` address here would be the local snapshot "
        "wearing the publication's clothes.",
    )
    source_id: str = Field(
        default="",
        description="The identifier the publisher assigns, and does not change when the "
        "document is edited or renamed. What makes an updated snapshot recognisable as the "
        "same document rather than a second one.",
    )
    version: str = Field(
        default="",
        description="The source's own version of this document, opaque and compared only for "
        "equality. A string rather than an integer because a source is entitled to version by "
        "revision hash, ETag or date, and parsing one into a number loses the others.",
    )
    created_at: datetime | None = Field(
        default=None, description="When the document was created at its source."
    )
    modified_at: datetime | None = Field(
        default=None,
        description="When the document was last edited **at its source**. Never this "
        "machine's ingestion time — see :attr:`LocalSnapshot.retrieved_at` and "
        "``documents.indexed_at``, which are the other two and are different questions.",
    )
    content_type: str = Field(
        default="",
        description="The media type the source published, as ``type/subtype``. Worth carrying "
        "separately from the snapshot's own media type: a page served as ``text/html`` may be "
        "mirrored to a file whose suffix says something else, and the local suffix is the "
        "weaker evidence.",
    )
    section_path: tuple[str, ...] = Field(
        default=(),
        description="Where the document sits in its source's hierarchy, coarsest first — a "
        "space and its parent pages, a manual and its chapter. The document's own title is "
        "**not** part of it; the chunker appends that itself, and a path carrying it twice "
        "reaches the embedder as emphasis nobody intended.",
    )

    @model_validator(mode="after")
    def _validated(self) -> Self:
        """Every declared field, checked. Raises :class:`ValueError` naming what was wrong.

        One validator rather than a decorator per field so that the message can name the field
        it came from — a manifest author reading "invalid" learns nothing, and this is the
        error they will actually see.
        """
        _printable(self.title, field="title")
        _printable(self.source_id, field="source_id")
        _printable(self.version, field="version")
        _aware(self.created_at, field="created_at")
        _aware(self.modified_at, field="modified_at")
        self._validate_canonical_uri()
        self._validate_content_type()
        self._validate_section_path()
        return self

    def _validate_canonical_uri(self) -> None:
        if not self.canonical_uri:
            return
        _printable(self.canonical_uri, field="canonical_uri")
        parsed = urlsplit(self.canonical_uri)
        scheme = parsed.scheme.lower()
        if scheme not in CITABLE_SCHEMES:
            msg = (
                f"canonical_uri {self.canonical_uri!r} uses the {scheme or 'absent'!r} scheme; "
                f"a citable address is one of {sorted(CITABLE_SCHEMES)}. A local path is not a "
                f"canonical address — manicule records the snapshot's location itself."
            )
            raise ValueError(msg)
        if not parsed.hostname:
            msg = (
                f"canonical_uri {self.canonical_uri!r} names no host, so it addresses nothing a "
                f"reader could open"
            )
            raise ValueError(msg)

    def _validate_content_type(self) -> None:
        if self.content_type and not _MEDIA_TYPE.match(self.content_type):
            msg = (
                f"content_type {self.content_type!r} is not a ``type/subtype`` media type, e.g. "
                f"'text/html'"
            )
            raise ValueError(msg)

    def _validate_section_path(self) -> None:
        if len(self.section_path) > MAX_SECTION_DEPTH:
            msg = (
                f"section_path is {len(self.section_path)} deep, over the "
                f"{MAX_SECTION_DEPTH}-element limit"
            )
            raise ValueError(msg)
        for index, element in enumerate(self.section_path):
            if not _printable(element, field=f"section_path[{index}]"):
                msg = f"section_path[{index}] is empty; a hierarchy has no anonymous levels"
                raise ValueError(msg)

    @model_validator(mode="after")
    def _says_something(self) -> Self:
        """A record that identifies nothing is refused rather than stored.

        The three fields checked here are the ones that make a citation better than the
        filename it replaces. A record carrying only, say, a ``created_at`` would take the
        canonical-metadata path through every surface and then render exactly what it would
        have rendered without one — a silent partial success, and the reader has no way to tell
        that the manifest they wrote is doing nothing.
        """
        if not (self.title or self.canonical_uri or self.source_id):
            msg = (
                "a source record must supply at least one of title, canonical_uri or "
                "source_id; without one of those it says nothing a local filename does not"
            )
            raise ValueError(msg)
        return self

    @property
    def section_elements(self) -> tuple[str, ...]:
        """:attr:`section_path`, for the breadcrumb the chunker builds.

        A named accessor rather than the tuple itself because this is the *only* thing that
        propagates from the document's source record into what an embedder reads, and the chunk
        does not copy it — see :meth:`manicule.core.content.Document.provenance`.
        """
        return self.section_path


class LocalSnapshot(BaseModel):
    """This machine's copy: where it sits, and when it was taken.

    Two of these fields are observations and one is a claim, and the difference is stated
    because nothing downstream could otherwise tell:

    :attr:`path` is observed.
        manicule fills it in from the artefact it actually read, relative to the ingestion
        root. **A manifest never populates it.** A manifest may *declare* a snapshot path, and
        that declaration is compared against this one and refused if it disagrees — so a
        manifest can contradict what is on disk, and can never relocate it. That is the
        difference between validating a path and not dereferencing one, and only the second is
        a defence.

    :attr:`retrieved_at` is declared.
        When the mirror was taken is not knowable from the file: a modification time moves when
        a file is copied, restored or checked out, so it is evidence about this filesystem and
        not about the retrieval. Whoever made the snapshot is the only party that knows, so it
        is theirs to state and it is marked as theirs.

    **There is no checksum field, deliberately.** ``documents.content_hash`` already holds the
    digest of exactly these bytes, computed by the pipeline over what the connector delivered.
    A second copy here would be a second authority for one fact, load-bearing precisely when
    the two disagreed, and citations read the column instead. A manifest's *declared* checksum
    is cross-checked at ingest against that same column and never stored beside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(
        default="",
        description="Where the snapshot sits, relative to the configured ingestion root. "
        "Relative rather than absolute so that a citation reproduces on another machine and "
        "does not publish this one's directory layout.",
    )
    retrieved_at: datetime | None = Field(
        default=None,
        description="When this copy was taken from the source, as whoever took it declared. "
        "Distinct from the source's own modification time and from manicule's indexing time; "
        "all three are separate questions and none substitutes for another.",
    )

    @model_validator(mode="after")
    def _validated(self) -> Self:
        _printable(self.path, field="snapshot path")
        _aware(self.retrieved_at, field="retrieved_at")
        return self


class Provenance(BaseModel):
    """Where a document came from, in both senses at once.

    Exactly one of :attr:`source` and :attr:`unavailable_reason` is set, on the same principle
    as :class:`~manicule.core.content.Retention` and
    :class:`~manicule.core.anchors.Unlocated`: absent **with a stated reason**, visible in
    diagnostics, never a silent partial success. A manifest that was found and refused is a
    thing an operator needs to be told about — it is usually a typo, and the symptom otherwise
    is a citation that quietly names a file.

    A document with no manifest at all carries **no record whatever**, not an empty one. That is
    what keeps ordinary local files working exactly as they did: nothing is written, nothing is
    read, and no code path they take is new.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SourceMetadata | None = Field(
        default=None, description="The publication's own account of itself, validated."
    )
    snapshot: LocalSnapshot | None = Field(
        default=None,
        description="This machine's copy. Absent for a document fetched over a network, which "
        "has no local snapshot to describe — the record is still worth having, because the "
        "canonical half of it is what a citation renders.",
    )
    unavailable_reason: str = Field(
        default="",
        description="Why there is no :attr:`source`, in words an operator can act on. Set "
        "exactly when a record was attempted and refused.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.source is None) == (not self.unavailable_reason):
            msg = (
                "a Provenance carries either a source record or a reason it has none; got "
                f"source={self.source!r} unavailable_reason={self.unavailable_reason!r}"
            )
            raise ValueError(msg)
        return self

    @property
    def usable(self) -> bool:
        """Whether this record can improve a citation."""
        return self.source is not None

    def as_metadata_value(self) -> dict[str, JsonValue]:
        """This record, JSON-shaped, for storing under :data:`PROVENANCE_KEY`.

        ``exclude_none`` is off for the same reason
        :meth:`manicule.app.results.Envelope.as_json` leaves it off: a reader checking a key's
        presence and a reader checking its value should not get different answers.
        """
        return cast("dict[str, JsonValue]", self.model_dump(mode="json"))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Provenance | None:
        """The record in ``metadata``, validated, or ``None`` when there is not a usable one.

        ``None`` covers three cases that are all the same answer downstream — no key, a key
        holding something that is not an object, and a key holding an object that does not
        validate. Every one of them means "no authoritative metadata", and the caller's
        fallback is the local filename and URI, which is what it would have used anyway.

        **Failing closed is the point.** A record that has been through a database is input
        again, and the alternative to returning ``None`` is raising — which would make one
        malformed row break every listing that touches it. The write path is where a bad
        manifest gets its diagnostic (:attr:`unavailable_reason`); this path's job is to be
        impossible to poison.
        """
        raw = metadata.get(PROVENANCE_KEY)
        if not isinstance(raw, Mapping):
            return None
        try:
            return cls.model_validate(dict(cast("Mapping[str, object]", raw)))
        except ValueError:
            # Not logged from here: `from_metadata` runs once per document per listing, and a
            # corrupt row would emit a line per render. `doctor` is where a corpus-wide
            # question about stored records belongs.
            return None


__all__ = [
    "CITABLE_SCHEMES",
    "MAX_FIELD_CHARS",
    "MAX_SECTION_DEPTH",
    "PROVENANCE_KEY",
    "LocalSnapshot",
    "Provenance",
    "SourceMetadata",
]
