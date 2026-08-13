"""Enriched standalone HTML: separating a page's identity and its body from the wrapper around them.

Some offline exporters write one HTML file per page. The file holds three things at once — a
machine-readable section stating the page's own identity, a wrapper around the *original*
storage-format XHTML, and ordinary HTML around both. Ingested generically, all three become one
document: the metadata banner is indexed as prose, the storage body reaches the HTML parser
instead of the one that understands ``ac:`` and ``ri:``, and a code macro's language, a panel's
severity and a Graphviz engine end up in the vector as things the page says.

This module is the **adapter**: given the file's text and a set of profiles, it answers with the
page's validated identity and the storage body on its own, or it refuses and names why. It writes
nothing, opens nothing and fetches nothing. :mod:`.enriched_html` uses it to write manifests and
:mod:`.filesystem` uses it at fetch, so the two cannot come to disagree about what an enriched
page is.

**A profile, never a heuristic.** :class:`EnrichedProfile` states the four things that vary
between exporters — how the metadata section marks itself, how the storage body marks itself,
what representation the body is, and which label spellings the exporter uses — and nothing here
matches anything a profile did not name. That is the difference between reading an exporter's
output and guessing at a document's shape: a bare ``<main>`` is not a storage body, a ``<section>``
is not metadata, and a file that matches no profile is *reported as unsupported* rather than
adapted on a hunch. :func:`EnrichedProfile._marked` refuses a selector that is not an attribute
selector for exactly this reason.

**Ambiguity is a refusal.** Exactly one metadata section and exactly one storage body, or the
file is refused naming what it had instead. Two candidate bodies is not a case with a sensible
default: the two contain different text, whichever is chosen is chosen silently, and a corpus
indexing the wrong half of a page reports nothing at all. :data:`OUTCOMES` writes the whole
decision out as a table rather than leaving it distributed through the branches, because the one
thing worse than an unenumerated rule is one whose gaps nobody has looked at.

**Everything read here is untrusted.** The file is inside somebody's corpus, so its author is
anyone with write access to the wiki it came from — or to the directory. Three consequences,
each handled by construction rather than by a check somebody has to keep correct:

*No value read out of the document reaches the filesystem.* This module returns strings. It has
    no output path, no root, and no file handle. A page id of ``../../../etc/cron.d/x`` is a
    string in a field.

*Nothing is fetched or executed.* A canonical address is read out of an ``href`` and validated
    as a citable URI; there is no network client here to dereference it with. Scripts are not
    run because nothing renders, and macro bodies stay text — see the note on CDATA below.

*Validation is the real model's.* Extracted fields construct a
    :class:`~manicule.core.provenance.SourceMetadata`, so a ``javascript:`` canonical URI, a
    control character in a title and a naive timestamp are refused *here* by exactly the code
    that would refuse them at ingest. A second implementation of those rules would be a second
    thing to keep in step, and the two would disagree the first time either was edited.

**Why the body is prepared before it is cut out.** :func:`~manicule.parsers.web.recover_cdata`
and :func:`~manicule.parsers.confluence.close_empty_elements` run over the whole file *before* it
is parsed, for the same reasons the storage parser runs them over its own input: an HTML engine
reparses ``<![CDATA[…]]>`` as a bogus comment and deletes the body of every code, ``noformat``
and Graphviz macro, and it does not honour self-closing syntax on ``<ri:page …/>``, so every
following sibling becomes that element's child. Cutting the body out of a tree built without them
would preserve a document that had already lost its macro bodies. Running them first means the
extracted body carries the recovered text, escaped, and re-running them on it downstream is a
no-op — which is what makes the DOT source come back character for character.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.ids import content_hash
from manicule.core.provenance import SourceMetadata
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE

if TYPE_CHECKING:
    from selectolax.lexbor import LexborHTMLParser, LexborNode

__all__ = [
    "ADAPTER_VERSION",
    "DEFAULT_LABELS",
    "DEFAULT_PROFILE",
    "ENRICHED_KEY",
    "MAX_BODY_BYTES",
    "MAX_HTML_BYTES",
    "METADATA_SELECTOR",
    "OUTCOMES",
    "REPRESENTATIONS",
    "STORAGE_SELECTOR",
    "Adaptation",
    "AdapterOutcome",
    "EnrichedPage",
    "EnrichedProfile",
    "UnusablePageError",
    "adapt",
    "extract",
]

ADAPTER_VERSION: Final = "enriched-html/1"
"""Identity of this extraction, recorded on every page it adapts.

**A version rather than a flag, because the derived body is a product of code and has to be
rebuildable when that code changes.** The rule the storage parser's ``parse_fp`` follows one
layer up (``docs/storage.md`` §6.4): what was made of the bytes is as much a part of a document's
lineage as the bytes themselves, and a corpus holding two generations of extracted body with
nothing to tell them apart is the failure that field exists to end. It travels in the change
token, so bumping it re-adapts and re-parses every enriched page without anything being
re-downloaded — the snapshot is a local file and was never thrown away.
"""

ENRICHED_KEY: Final = "enriched_adaptation"
"""Document-metadata key recording what the adapter did to this page.

**Here rather than on** :class:`~manicule.core.provenance.LocalSnapshot`, and the choice is the
one ``docs/storage.md`` §4.2.1 already settled for ``space_key``: the provenance record carries
what *every* source has, and anything one situation means and others do not stays in the
connector's own keys. A snapshot checksum distinct from the document's content hash is exactly
that — it exists only where a document's bytes are *derived from* a local file rather than being
it, which is true of an enriched export and of nothing else manicule ingests. Adding a field to
``LocalSnapshot`` would have put a checksum on every document that has no second set of bytes to
be a checksum of, and its docstring's argument against one — that ``documents.content_hash``
already holds the digest of exactly these bytes — remains true for all of them.

The record holds the six facts §1 of the specification asks be distinguishable, minus the two the
core already stores: the local snapshot path is :attr:`LocalSnapshot.path` and the extracted
body's digest is ``documents.content_hash``. Both are repeated here anyway, because a diagnostic
that had to join three places to answer "what was extracted from what" would not be run.
"""

MAX_HTML_BYTES: Final = 8 * 1024 * 1024
"""Largest enriched page this will read.

Generous for a wiki page, and present because without it a file in a directory somebody pointed
the connector at decides how much memory the run allocates.
"""

MAX_BODY_BYTES: Final = MAX_HTML_BYTES
"""Largest extracted body this will hand on.

Bounded separately from the input rather than assumed to follow from it. The two are equal today
and the *reason* they are two constants is that they bound different things: the input bound is
about how much is read, and this one is about how much is produced — and an extraction is a
transformation, so "the output cannot be bigger than the input" is a property of the current
implementation rather than a promise the interface makes.
"""

METADATA_SELECTOR: Final = "[data-source-metadata]"
"""How the default profile's metadata section identifies itself.

An attribute rather than a class or a heading, because it is the one part of these documents
addressed to a machine. A class is styling and an exporter is entitled to change it; a heading is
prose and is translated.
"""

STORAGE_SELECTOR: Final = '[data-document-representation="storage"]'
"""How the default profile's storage body identifies itself.

The attribute names the *representation*, not the element, which is why a bare ``main`` is not
this selector and never will be. An exporter that wraps its body in a ``<div>`` is still saying
the same thing about it, and an exporter whose ``<main>`` holds a rendered page rather than
storage format is saying something different with the same element.
"""

REPRESENTATIONS: Final[Mapping[str, str]] = {
    CONFLUENCE_MEDIA_TYPE: "Confluence storage format, read by ConfluenceStorageParser",
}
"""Representations a profile may declare, and what each routes to.

**An allowlist, and a table rather than a syntax check.** A profile's representation decides
which parser untrusted extracted markup is handed to, so "any well-formed media type" would let a
configuration file aim a hostile document at any parser installed — and a *misspelled* one would
route to no parser at all, leaving the page in ``unsupported_media_type`` while the configuration
looked correct. One entry today. Adding a representation is one row plus the parser that claims
it, and a profile naming anything else is refused with this list in the message.
"""

_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ancestors",
        "canonical_uri",
        "created_at",
        "modified_at",
        "retrieved_at",
        "source_id",
        "space",
        "title",
        "version",
    }
)
"""Every field a label may fill. A profile naming anything else is refused at construction.

Refused rather than ignored, on :mod:`.sidecar`'s reasoning about unknown manifest keys: the
overwhelmingly likely unknown field name is a misspelling of a known one, and ignoring it means
the configured alias silently does nothing — indistinguishable from not having written it, which
is the failure the alias was added to fix.
"""

DEFAULT_LABELS: Final[Mapping[str, str]] = {
    "page id": "source_id",
    "id": "source_id",
    "title": "title",
    "space": "space",
    "space key": "space",
    "ancestors": "ancestors",
    "parent": "ancestors",
    "parents": "ancestors",
    "version": "version",
    "last modified": "modified_at",
    "modified": "modified_at",
    "updated": "modified_at",
    "created": "created_at",
    "source": "canonical_uri",
    "canonical url": "canonical_uri",
    "canonical": "canonical_uri",
    "url": "canonical_uri",
    "retrieved": "retrieved_at",
    "exported": "retrieved_at",
}
"""Labels the default profile understands, normalised, mapped to the field each fills.

Several spellings per field because these documents are written by exporters that were not
coordinating with each other, and "Last modified", "Modified" and "Updated" are the same fact.
Unknown *labels* are ignored rather than refused — an exporter adding a row of its own is not an
error, and the failure that has to be caught is the absence of an identity, which is checked
directly. That is the opposite of the rule for unknown *fields* in :data:`_FIELDS`, and the two
differ because one is somebody else's document and the other is this installation's own
configuration.
"""

_ANCESTOR_SEPARATORS: Final = ("\u203a", "\u00bb", ">", "/", ",")
"""How exporters write a breadcrumb. Tried in order; the first one present splits the value.

The first two are U+203A and U+00BB, written as escapes because they are confusable with an ASCII
``>`` on sight — and that is what makes them worth listing. A breadcrumb rendered with a
typographic separator is a different byte from one rendered with the ASCII character, and an
exporter using the pretty one would otherwise fall through to the ``/`` rule and have its page
path split on the slashes inside a title.
"""

_ATTRIBUTE_SELECTOR = re.compile(
    r"""^[A-Za-z0-9_*.:#-]*                          # an optional element or class qualifier
        \[[A-Za-z_][A-Za-z0-9_.:-]*                  # the attribute name
        (?:[~|^$*]?=(?:"[^"]*"|'[^']*'|[^\]\s"']+))? # an optional value test
        \]$""",
    re.VERBOSE,
)
"""A selector that selects on an attribute, optionally qualified by an element.

Checked rather than assumed, because it is the whole of "a profile, never a heuristic". A
selector of ``main`` or ``section`` would make every such element on every page a candidate, and
the spec this implements forbids exactly that. An attribute is a statement an exporter made on
purpose; an element name is a fact about HTML.

Spelled out to the value test rather than left as "brackets with something in them", so a
configured ``[a=]`` is refused *here* — as configuration, at startup, naming the setting — rather
than reaching an HTML engine mid-walk and failing as though a document were at fault.
"""


class UnusablePageError(Exception):
    """A page was found and cannot be adapted. The message is what an operator reads.

    Carries :attr:`outcome` so a diagnostic can count refusals by kind without matching on prose.
    The message and the outcome are set together at every raise site, which is what stops a
    report's counters from describing a different failure than its reasons do.
    """

    def __init__(self, detail: str, outcome: AdapterOutcome) -> None:
        super().__init__(detail)
        self.outcome = outcome


class AdapterOutcome(StrEnum):
    """What became of one considered file.

    Every file a run looks at ends on exactly one of these, which is what makes "a run that
    adapted zero pages says so" a property of the type rather than a sentence somebody remembered
    to write. :attr:`NO_PROFILE` is the only one that is not a complaint: a directory of ordinary
    HTML produces nothing but those, and that is a correct and complete answer.
    """

    ADAPTED = "adapted"
    NO_PROFILE = "no_profile"
    INVALID_METADATA = "invalid_metadata"
    MISSING_BODY = "missing_body"
    AMBIGUOUS = "ambiguous"
    DUPLICATE_IDENTITY = "duplicate_identity"
    ALREADY_PRESENT = "already_present"
    """A page that adapts cleanly and whose manifest was left alone.

    Its own value rather than folded into :attr:`NO_PROFILE`, which is where it was and which was
    a plain untruth: a second conversion run over a converted directory reported
    ``no_profile: 2`` — "neither of these is an enriched page" — about a directory in which one of
    them demonstrably was. The two answers lead opposite ways. ``no_profile`` says *point this at
    a different directory*; this says *pass ``--force`` if you meant to replace what is there*.
    Found by running the command twice and reading what it said.
    """

    FAILED = "failed"


NONE: Final = "0"
ONE: Final = "1"
MANY: Final = "many"
"""How many of a profile's markers a file carries. Three buckets, because the fourth answer a
count could give — *which* of several — is the one this refuses to have an opinion about."""

OUTCOMES: Final[Mapping[tuple[str, str], AdapterOutcome]] = {
    # (metadata sections matched, storage bodies matched)
    (NONE, NONE): AdapterOutcome.NO_PROFILE,
    (NONE, ONE): AdapterOutcome.INVALID_METADATA,
    (NONE, MANY): AdapterOutcome.AMBIGUOUS,
    (ONE, NONE): AdapterOutcome.MISSING_BODY,
    (ONE, ONE): AdapterOutcome.ADAPTED,
    (ONE, MANY): AdapterOutcome.AMBIGUOUS,
    (MANY, NONE): AdapterOutcome.AMBIGUOUS,
    (MANY, ONE): AdapterOutcome.AMBIGUOUS,
    (MANY, MANY): AdapterOutcome.AMBIGUOUS,
}
"""Every combination of marker counts, and what each one is. Nine rows, none of them implied.

**Written out rather than expressed as conditions, and doing so found two cases the conditions
had hidden.** The first was ``(NONE, ONE)``: a file carrying a storage body and no metadata
section at all. A predicate asking "did this profile engage?" answers *yes*, then the metadata
extraction runs against nothing and the page is refused for declaring no id — a message that
sends the reader looking for a missing row inside a section that is not there. It is a page that
states no identity of its own, and it says so now.

The second was ``(MANY, NONE)``. Both counts are wrong, and a rule that checked bodies first
called it :attr:`~AdapterOutcome.MISSING_BODY` — which is a repair instruction ("add a body") for
a file whose actual problem is that it is two pages concatenated. Multiplicity is the complaint
that cannot be fixed by looking at one page, so it wins.

``(NONE, NONE)`` is the only row that is not a complaint: a profile matched nothing, so the next
one gets a turn, and a file matching none of them is ordinary HTML rather than a broken enriched
page. A directory of ordinary HTML produces nothing but that row.
"""


class EnrichedProfile(BaseModel):
    """One exporter's conventions: how it marks its metadata, its body, and what the body is.

    Validated configuration rather than code, so a site whose exporter spells the markers
    differently adds a profile instead of a patch. Every field is checked at construction, and the
    checks are the ones that keep a profile from becoming a heuristic — see
    :data:`_ATTRIBUTE_SELECTOR` and :data:`REPRESENTATIONS`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        description="What this profile is called. Recorded on every page it adapts, so a corpus "
        "can be asked which exporter's convention produced it.",
    )
    metadata_selector: str = Field(
        default=METADATA_SELECTOR,
        description="How the page's metadata section marks itself. Must select on an attribute.",
    )
    body_selector: str = Field(
        default=STORAGE_SELECTOR,
        description="How the embedded body marks itself. Must select on an attribute.",
    )
    representation: str = Field(
        default=CONFLUENCE_MEDIA_TYPE,
        description="What the embedded body is, as the media type it will be parsed under.",
    )
    labels: Mapping[str, str] = Field(
        default=DEFAULT_LABELS,
        description="Label spellings this exporter uses, normalised, mapped to the field each "
        "fills. Replaces the defaults rather than extending them, so a profile states its whole "
        "vocabulary and a reader does not have to know what it inherited.",
    )

    @model_validator(mode="after")
    def _validated(self) -> Self:
        self._marked(self.metadata_selector, field="metadata_selector")
        self._marked(self.body_selector, field="body_selector")
        if self.representation not in REPRESENTATIONS:
            known = ", ".join(sorted(REPRESENTATIONS))
            msg = (
                f"profile {self.name!r} declares representation {self.representation!r}, which "
                f"nothing here parses. A profile's representation decides which parser untrusted "
                f"extracted markup reaches, so it is an allowlist: {known}"
            )
            raise ValueError(msg)
        unknown = sorted(set(self.labels.values()) - _FIELDS)
        if unknown:
            msg = (
                f"profile {self.name!r} maps labels to {unknown}, which name no field. A label "
                f"fills one of {sorted(_FIELDS)}; a misspelling here silently does nothing, "
                f"which is indistinguishable from not having configured the alias at all"
            )
            raise ValueError(msg)
        return self

    @staticmethod
    def _marked(selector: str, *, field: str) -> None:
        """Refuse a selector that would make an ordinary element a marker.

        Raises:
            ValueError: The selector does not select on an attribute. ``main`` matches every
                ``<main>`` on every page, which is the heuristic this whole mechanism exists
                instead of — and the failure would be a page's navigation indexed as its body,
                silently, on a corpus where it happened to be the only ``<main>``.
        """
        if not _ATTRIBUTE_SELECTOR.match(selector.strip()):
            msg = (
                f"{field} {selector!r} does not select on an attribute. An enriched document is "
                f"recognised by a marker its exporter wrote on purpose, never by an element name "
                f"— '[data-source-metadata]' and 'main[data-document-representation=\"storage\"]' "
                f"are markers; 'main' and 'section' are facts about HTML"
            )
            raise ValueError(msg)


DEFAULT_PROFILE: Final = EnrichedProfile(name="standalone-storage")
"""The safe default: the form documented in ``docs/connectors/enriched-html.md``.

Named for what it is rather than for any product, because the convention is "one standalone file,
metadata section, storage body" and several exporters could satisfy it.
"""


@dataclass(frozen=True, slots=True)
class EnrichedPage:
    """One page's declared identity, validated.

    Two fields rather than one flat record, mirroring
    :class:`~manicule.core.provenance.Provenance`'s own split: :attr:`source` is what the
    publication says about itself, and :attr:`retrieved_at` is a statement about this copy. They
    are kept apart here for the reason they are kept apart there — the second is not a fact about
    the document, and a model that could hold both would let a caller write one where the other
    belonged.
    """

    source: SourceMetadata
    retrieved_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Adaptation:
    """One enriched page, taken apart: what it says it is, and the body worth indexing.

    The two checksums are the point of carrying both. :attr:`snapshot_checksum` digests the whole
    file as it sits on disk — the immutable local snapshot, which is what an audit reproduces
    from. :attr:`body_checksum` digests what was extracted from it, which is what the index is
    actually built out of. They answer different questions, they move independently (an exporter
    can rewrite a page's navigation without touching its body), and a record holding only one of
    them cannot say which of the two changed.
    """

    profile: EnrichedProfile
    page: EnrichedPage
    body: str
    snapshot_checksum: str
    body_checksum: str

    @property
    def representation(self) -> str:
        """The media type the extracted body is parsed under."""
        return self.profile.representation


def adapt(html: str, *, profiles: tuple[EnrichedProfile, ...] = (DEFAULT_PROFILE,)) -> Adaptation:
    """Take ``html`` apart, or refuse naming what was wrong with it.

    The first profile that **engages** decides, and engaging means matching either selector at
    least once. A profile that engages and is then ambiguous is a refusal rather than a fall
    through to the next: a document carrying two of one exporter's body markers must not be read
    as some other exporter's page, which is precisely what falling through would allow a hostile
    file to arrange.

    Args:
        html: The file's text. Parsed, never executed, never written back.
        profiles: The configured conventions, in precedence order.

    Returns:
        The page's validated identity and the body on its own, with a digest of each of the two
        things that can change independently.

    Raises:
        UnusablePageError: No profile engaged; or one did and the file has no body, several
            bodies, several metadata sections, no page id, or a value the citation interface will
            not carry. Every message names what was found, because "invalid page" tells whoever
            runs the exporter nothing about which line to look at.
    """
    from selectolax.lexbor import LexborHTMLParser  # noqa: PLC0415 - see the module docstring

    from manicule.parsers.confluence import close_empty_elements  # noqa: PLC0415
    from manicule.parsers.web import recover_cdata  # noqa: PLC0415

    # Both, and before the tree is built. Cutting a body out of a tree parsed without them would
    # preserve a document whose macro bodies an HTML engine had already deleted.
    tree = LexborHTMLParser(close_empty_elements(recover_cdata(html)))
    for profile in profiles:
        sections = tree.css(profile.metadata_selector)
        bodies = tree.css(profile.body_selector)
        outcome = OUTCOMES[(_bucket(len(sections)), _bucket(len(bodies)))]
        if outcome is AdapterOutcome.NO_PROFILE:
            continue
        if outcome is not AdapterOutcome.ADAPTED:
            raise _refusal(profile, outcome, sections=len(sections), bodies=len(bodies))
        return _adapted(profile, html=html, section=sections[0], body=bodies[0], tree=tree)

    named = ", ".join(repr(profile.name) for profile in profiles) or "none configured"
    msg = (
        f"matches no configured enriched-document profile ({named}), so it is an ordinary HTML "
        f"file as far as this is concerned and is left to generic ingestion"
    )
    raise UnusablePageError(msg, AdapterOutcome.NO_PROFILE)


def extract(
    html: str, *, profiles: tuple[EnrichedProfile, ...] = (DEFAULT_PROFILE,)
) -> EnrichedPage:
    """Just the page's declared identity, for a caller with no use for the body.

    :func:`adapt` without the extraction, and it refuses everything :func:`adapt` refuses — which
    is deliberate rather than incidental. A conversion that accepted a page whose body could not
    be isolated would write a manifest promising an identity for a document the connector will
    then decline to adapt, and the operator would have two reports disagreeing about one file.
    """
    return adapt(html, profiles=profiles).page


def _adapted(
    profile: EnrichedProfile,
    *,
    html: str,
    section: LexborNode,
    body: LexborNode,
    tree: LexborHTMLParser,
) -> Adaptation:
    """One matched page, validated and cut apart.

    Raises:
        UnusablePageError: The metadata declares no page id, declares one label twice with two
            values, carries a value the citation interface refuses, or the extracted body is over
            :data:`MAX_BODY_BYTES`.
    """
    fields = _fields(section, labels=profile.labels)
    source_id = fields.get("source_id", "")
    if not source_id:
        msg = (
            "declares no page id. That is the identifier an updated snapshot is recognised by, "
            "and without it a re-export is a second document rather than a new version of one"
        )
        raise UnusablePageError(msg, AdapterOutcome.INVALID_METADATA)

    try:
        metadata = SourceMetadata(
            title=fields.get("title") or _title(tree),
            canonical_uri=fields.get("canonical_uri", ""),
            source_id=source_id,
            version=fields.get("version", ""),
            created_at=_timestamp(fields.get("created_at"), field="created"),
            modified_at=_timestamp(fields.get("modified_at"), field="last modified"),
            content_type=profile.representation,
            section_path=_section_path(fields),
        )
    except ValueError as exc:
        msg = f"declares metadata this index will not cite ({_collapse(str(exc))})"
        raise UnusablePageError(msg, AdapterOutcome.INVALID_METADATA) from exc

    # The **outer** HTML of the matched element, not its contents. The wrapper is one unknown
    # element to the storage parser's walk, which recurses through it and reads what is inside —
    # so keeping it costs a container and saves reassembling a fragment from its children, where
    # a serialisation slip would be a silently truncated page rather than a failure.
    extracted = body.html or ""
    encoded = len(extracted.encode("utf-8"))
    if encoded > MAX_BODY_BYTES:
        msg = (
            f"extracts a {encoded}-byte body, over the {MAX_BODY_BYTES}-byte limit. The bound is "
            f"on what is produced as well as on what is read, because an extraction is a "
            f"transformation and its output size is not the input's"
        )
        raise UnusablePageError(msg, AdapterOutcome.FAILED)

    return Adaptation(
        profile=profile,
        page=EnrichedPage(
            source=metadata,
            retrieved_at=_timestamp(fields.get("retrieved_at"), field="retrieved"),
        ),
        body=extracted,
        snapshot_checksum=content_hash(html),
        body_checksum=content_hash(extracted),
    )


def _bucket(count: int) -> str:
    return NONE if count == 0 else ONE if count == 1 else MANY


_COMPLAINTS: Final[Mapping[AdapterOutcome, str]] = {
    AdapterOutcome.INVALID_METADATA: (
        "declares a storage body and no metadata section, so it states no identity of its own "
        "and there is nothing to record that its filename does not already say"
    ),
    AdapterOutcome.MISSING_BODY: (
        "declares metadata and no storage body, so there is nothing here to index as storage "
        "format. The page's text may be present as ordinary HTML, which is a different document"
    ),
    AdapterOutcome.AMBIGUOUS: (
        "carries more than one of a profile's markers. Which of several describes the page is "
        "exactly the ambiguity this cannot guess at, and choosing would index half a page and "
        "report success"
    ),
}
"""What to say first, per refusal. Keyed off the table, so a row added there needs one here."""


def _refusal(
    profile: EnrichedProfile, outcome: AdapterOutcome, *, sections: int, bodies: int
) -> UnusablePageError:
    """The refusal for a file that engaged a profile and could not be read from it.

    Names **both** counts however only one of them is wrong. Reporting the offending count alone
    reads as though the other were fine, and the two commonest shapes — an export that
    concatenated two pages into one file, and one that emitted a body with no metadata — are told
    apart precisely by the number that was *not* complained about.
    """
    msg = (
        f"{_COMPLAINTS[outcome]} (profile {profile.name!r}: {sections} x "
        f"{profile.metadata_selector}, {bodies} x {profile.body_selector})"
    )
    return UnusablePageError(msg, outcome)


def _fields(section: LexborNode, *, labels: Mapping[str, str]) -> dict[str, str]:
    """The labelled values in ``section``, keyed by the field each fills.

    Raises:
        UnusablePageError: One label appeared twice with two different values. Refused rather
            than resolved by order, because "the first one wins" is a rule nobody writing an
            exporter knows about, and the two values are equally likely to be the right one.
    """
    found: dict[str, tuple[str, str]] = {}
    for label, node, without in _rows(section):
        field = labels.get(label)
        if field is None:
            continue
        value = _value(node, without=without, prefer_href=field == "canonical_uri")
        if not value:
            continue
        previous = found.get(field)
        if previous is not None and previous[1] != value:
            # The *field* is what collides, and two different labels can fill one — "Source" and
            # "Canonical URL" both name the address. Reporting "declares 'canonical url' twice"
            # when the other statement was spelled "Source" sends the reader looking for a
            # duplicate row that is not there, so both labels are named.
            first_label, first_value = previous
            spelling = (
                f"{first_label!r}" if first_label == label else f"{first_label!r} and {label!r}"
            )
            msg = (
                f"declares {spelling} twice, as {first_value!r} and {value!r}. Which is the "
                f"page's is not something this can decide"
            )
            raise UnusablePageError(msg, AdapterOutcome.INVALID_METADATA)
        found[field] = (label, value)
    return {field: value for field, (_, value) in found.items()}


def _rows(section: LexborNode) -> Iterator[tuple[str, LexborNode, str]]:
    """``(normalised label, the node holding the value, text to strip)`` for every labelled row.

    The node rather than its text, because how a value should be read depends on which field it
    fills and only the caller knows that — see :func:`_value`.
    """
    for node in section.css("dt"):
        sibling = node.next
        while sibling is not None and sibling.tag == "-text":
            sibling = sibling.next
        if sibling is not None and sibling.tag == "dd":
            yield _label(node.text(deep=True)), sibling, ""
    for marker in section.css("strong, b, th"):
        parent = marker.parent
        if parent is None:
            continue
        whole = _collapse(parent.text(deep=True))
        marked = _collapse(marker.text(deep=True))
        if not whole.startswith(marked):
            continue
        yield _label(marked), parent, marked


def _value(node: LexborNode, *, without: str = "", prefer_href: bool = False) -> str:
    """``node``'s value: its text, or an anchor's address for a field that holds one.

    ``prefer_href`` is per field and not a property of the markup, which is the correction this
    signature exists to make. A canonical link is written ``<a href="…">canonical page</a>`` at
    least as often as it is written out, and recording the words "canonical page" as the page's
    address would be a citation pointing at nothing — but preferring the ``href`` *everywhere*
    turns ``<strong>Title:</strong> <a href="…">Retry policy</a>`` into a title that is a URL, and
    a linked title is not a rare document.

    **Read, never dereferenced.** Nothing in this module opens a URL.
    """
    if prefer_href:
        for anchor in node.css("a"):
            href = (anchor.attributes.get("href") or "").strip()
            if href:
                return href
    text = _collapse(node.text(deep=True))
    return text[len(without) :].strip().lstrip(":").strip() if without else text


def _label(raw: str) -> str:
    return _collapse(raw).rstrip(":").strip().lower()


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _section_path(fields: Mapping[str, str]) -> tuple[str, ...]:
    """The page's place in its source's hierarchy, coarsest first.

    The space leads because it is the outermost container, then any declared ancestors. The page's
    own title is **not** appended: the chunker adds that itself, and a path carrying it twice
    reaches the embedder as emphasis nobody intended
    (:attr:`~manicule.core.provenance.SourceMetadata.section_path`).
    """
    path: list[str] = []
    space = fields.get("space", "").strip()
    if space:
        path.append(space)
    path.extend(_ancestors(fields.get("ancestors", "")))
    return tuple(path)


def _ancestors(raw: str) -> Iterator[str]:
    value = raw.strip()
    if not value:
        return
    for separator in _ANCESTOR_SEPARATORS:
        if separator in value:
            for part in value.split(separator):
                if part.strip():
                    yield part.strip()
            return
    yield value


def _timestamp(raw: str | None, *, field: str) -> datetime | None:
    """``raw`` as an aware datetime, or ``None`` when it was not declared.

    Raises:
        UnusablePageError: It is not a timestamp, or it carries no offset. A naive one is refused
            rather than assumed to be UTC: read as UTC it is wrong by the exporting host's offset,
            and it is used to decide which of two versions is newer.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"declares {field} {text!r}, which is not an ISO-8601 timestamp"
        raise UnusablePageError(msg, AdapterOutcome.INVALID_METADATA) from exc
    if parsed.tzinfo is None:
        msg = (
            f"declares {field} {text!r}, which carries no UTC offset and so does not say which "
            f"zone it is in"
        )
        raise UnusablePageError(msg, AdapterOutcome.INVALID_METADATA)
    return parsed


def _title(tree: LexborHTMLParser) -> str:
    element = tree.css("title")
    return _collapse(element[0].text(deep=True)) if element else ""
