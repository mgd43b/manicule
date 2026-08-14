"""Fingerprints — the identity of a process that produced stored data.

Four exist: :class:`~manicule.core.embedding.EmbedFingerprint` for vectors,
:class:`ChunkFingerprint` for chunk boundaries, :class:`ParseFingerprint` for the text
and anchors a parser extracted, and :class:`GlossaryFingerprint` for the definitions a
detector read out of chunks. All four are persisted alongside the index and compared
before anything is written, because all four describe transformations whose output is
useless when mixed with output from a different version of themselves — and useless in the
quiet way, where nothing raises and every answer is slightly wrong.

They share these semantics deliberately:

Comparison is on a canonical serialization, byte for byte
    Not on one field. A guard that compared only a dimension, or only a version number,
    passes exactly the cases that matter — two different models at the same dimension, two
    different grammars at the same chunk size.

Identity is a declared subset of the fields
    Some fields describe the producer without affecting its output. Those are recorded for
    diagnostics and excluded from comparison, so that moving between machines does not
    invalidate a corpus for no reason.

A mismatch raises, always
    Never a warning. There is nothing downstream that can detect mixed output, so the only
    place it can be caught is here.

Where they differ is scope, and it follows from what each one produces.
:class:`ChunkFingerprint` and ``EmbedFingerprint`` describe one process applied to a whole
corpus, so they are compared once per run and a mismatch refuses the run.
:class:`ParseFingerprint` describes one parser applied to one document, so it is compared per
document and a mismatch invalidates that document and no other. §6.4 of ``docs/storage.md``
calls that per-document lineage; parsing is the case where it is the *only* honest scope.
:class:`GlossaryFingerprint` is per document as well, for a different reason: there is one
detector rather than many, but its output is repaired one document at a time from chunks that
are already stored, so a mismatch has to name the documents rather than refuse the run.

**A mismatch does not raise for that last one, and it is the only exception.** The three above
describe data that is *incomparable* across versions — a vector from another model, chunks from
another budget — so mixing them is a defect nothing downstream can detect and the only available
answer is to stop. Glossary entries from a superseded detector are merely wrong, which is a
repairable state and one an operator has to be able to survey, re-index around and fix in place.
Refusing every run against a corpus whose detector has moved would make a detector fix
unshippable: the fix is what makes the corpus stale.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar, Final, Self, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from manicule.core.errors import FingerprintMismatchError

PROVISIONAL_TOKENIZER_PREFIX: Final = "provisional:"
"""What a token counter that is not the embedder's own puts in front of its identity.

The prefix lives here, beside the field it appears in, because two things read it and they
must not be able to disagree: :class:`~manicule.chunking.tokens.TokenCounter` stamps it on
every count it takes without a bound model, and :attr:`ChunkFingerprint.provisional` is
exactly the question "is this string stamped". Deriving the flag from the identity string
rather than storing it separately is what makes the two impossible to contradict — a boolean
field could be ``False`` beside a stamped id, and the refusal would then pass a corpus
measured with a stand-in vocabulary.

The whole of what follows the prefix is identity as well, and that is the point of §1.2 of
``docs/parsing.md``: a stand-in counter must name the vocabulary it stood in with *and* the
safety factor it inflated by, because both move every boundary.
"""

DETECTION_DISABLED: Final = "disabled"
"""What :attr:`GlossaryFingerprint.detector` says when detection was switched off.

A recorded value rather than an absent one, which is the whole of requirement 8. ``NULL``
already means "never recomputed", and if a disabled run also wrote nothing then the two states
would be one column value and an operator could not tell a corpus that has no definitions from
one whose definitions were never looked for. Turning detection back on changes the installed
fingerprint, so every document stamped this way is selected by the glossary repair on the next
survey — which is the behavior somebody switching the feature back on expects and would
otherwise have to know to ask for.
"""


class Fingerprint(BaseModel):
    """Base for fingerprints. Subclasses declare which of their fields are identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ()

    def identity(self) -> dict[str, JsonValue]:
        """The fields that decide comparability, as JSON-shaped values."""
        dumped: dict[str, JsonValue] = self.model_dump(mode="json")
        return {name: dumped[name] for name in self.IDENTITY_FIELDS}

    def canonical(self) -> str:
        """A stable serialization of :meth:`identity`.

        Keys sorted and separators fixed, so two runs of the same version produce byte-equal
        output and a stored fingerprint can be compared without being parsed.
        """
        return json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))

    def matches(self, other: Self) -> bool:
        """Whether ``other`` produces output interchangeable with this one's."""
        return type(self) is type(other) and self.canonical() == other.canonical()

    def changed_fields(self, other: Self) -> frozenset[str]:
        """Which identity fields differ.

        What makes selective invalidation possible: a grammar upgrade that changes one
        language should invalidate the documents in that language, not the corpus.
        """
        mine = self.identity()
        theirs = other.identity()
        return frozenset(name for name in (*mine, *theirs) if mine.get(name) != theirs.get(name))

    def require_match(self, other: Self) -> None:
        """Raise unless ``other`` is interchangeable with this one.

        Args:
            other: The fingerprint offered by whatever is about to read or write.

        Raises:
            FingerprintMismatchError: When they differ, naming the fields that differ.
        """
        if self.matches(other):
            return
        changed = ", ".join(sorted(self.changed_fields(other))) or "type"
        msg = (
            f"{type(self).__name__} mismatch on {changed}: the index was built with "
            f"{self.describe()}, but {other.describe()} was offered. Re-run against the "
            f"index this produced, or rebuild the index."
        )
        raise FingerprintMismatchError(msg)

    def describe(self) -> str:
        """A one-line human-readable form, for error messages and diagnostics."""
        return self.canonical()


class ChunkFingerprint(Fingerprint):
    """The identity of the process that decided where chunks begin and end.

    Persisted per corpus and recorded per document, so a change can be traced to the
    documents it affects. Re-chunking is cheaper than re-embedding but not free, and
    invalidating everything because one grammar moved is the expensive mistake.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "chunker",
        "version",
        "max_tokens",
        "overlap_tokens",
        "tokenizer_id",
        "grammars",
        "embed_text_middleware",
    )

    chunker: str = Field(min_length=1, description="Registered chunker that produced the chunks.")
    version: str = Field(min_length=1, description="The chunker's own version.")

    max_tokens: int = Field(
        gt=0,
        description="Token budget per chunk, measured on ``embed_text``. Must not exceed the "
        "embedder's ``max_sequence_length``: past that limit the text is truncated with no "
        "error, and the chunk is indexed as its first N tokens while still claiming all of "
        "its text — a citation that quotes words the index never saw. The chunker enforces "
        "this when it runs; re-embed does not re-chunk, so that path enforces it with "
        "``require_within_context`` instead.",
    )
    overlap_tokens: int = Field(
        default=0, ge=0, description="Tokens repeated between adjacent chunks."
    )
    tokenizer_id: str = Field(
        min_length=1,
        description="Which tokenizer counted the tokens, and — when it was not the "
        "embedder's own — what was done to the number afterwards. A budget is meaningless "
        "without it: the same text is a different number of tokens under a different "
        "vocabulary, and an inflated count moves every boundary again. A stand-in counter "
        "records the whole of that as one string, prefixed "
        f"``{PROVISIONAL_TOKENIZER_PREFIX}``, so that no part of it can vary behind an "
        "identifier that stays still.",
    )
    grammars: dict[str, str] = Field(
        default_factory=dict,
        description="Version by language for structure-aware chunking, e.g. "
        "``{'python': '0.21.0'}``. Recorded per language so that upgrading one grammar "
        "invalidates the documents in that language and leaves the rest alone.",
    )
    embed_text_middleware: tuple[str, ...] = Field(
        default=(),
        description="Sorted ``name@version`` for every middleware declaring "
        "``mutates_embedded_text``. Empty on a chunker's own fingerprint; the ingest "
        "pipeline folds the configured set in before comparing. Without it, two instances "
        "with identical configuration and different middleware produce different vectors "
        "from identical source bytes and **neither fingerprint refusal notices**, because "
        "neither otherwise knows middleware exists. Adding, removing or upgrading one is "
        "then exactly as loud as changing the chunk budget, which is what it is.",
    )

    @property
    def provisional(self) -> bool:
        """Whether these boundaries were measured without the model that will embed them.

        Read off :attr:`tokenizer_id` rather than carried in a field of its own, so the
        answer and the identity cannot disagree. Ingest refuses a provisional fingerprint
        outright (``docs/parsing.md`` §1.2): a stand-in vocabulary undercounts by an unknown
        margin, undercounting is the direction that truncates silently, and the safety factor
        that compensates is a guess rather than a measurement. Provisional chunks exist to be
        inspected, never to be served.
        """
        return self.tokenizer_id.startswith(PROVISIONAL_TOKENIZER_PREFIX)

    def with_middleware(self, declarations: Sequence[str]) -> ChunkFingerprint:
        """This fingerprint, with ``declarations`` folded into its identity.

        Sorted and de-duplicated here rather than at every call site, so the identity does
        not depend on the order configuration happened to list middleware in — which is a
        legitimate thing to change without changing a single vector.
        """
        return self.model_copy(update={"embed_text_middleware": tuple(sorted(set(declarations)))})

    @override
    def describe(self) -> str:
        mutating = (
            f", mutated by {', '.join(self.embed_text_middleware)}"
            if self.embed_text_middleware
            else ""
        )
        return (
            f"{self.chunker} {self.version} ({self.max_tokens}+{self.overlap_tokens} tokens, "
            f"{self.tokenizer_id}{mutating})"
        )


class ParseFingerprint(Fingerprint):
    """The identity of the process that produced one document's ``text`` and its anchors.

    ``docs/parsing.md`` §3.0 says ``text`` is immutable after parse. That is true within a
    parser version and quietly false across one: the same bytes reduce to different text
    after ``pypdfium2`` moves, and because change detection keys on the connector's source
    bytes, nothing already stored is ever re-read while newly ingested identical bytes parse
    differently. A corpus then holds two generations of extracted text with no column saying
    which is which.

    **Anchors are the sharper edge.**
    :func:`~manicule.testing.contracts.assert_parser_contract` checks that an anchor resolves
    to the text its block claims *at parse time*, and says nothing about an anchor stored
    under a previous version. A library that changes an offset or coordinate convention
    leaves old anchors resolving into text produced differently — a plausible, wrong
    location, which is the defect this project exists not to reproduce.

    **Why this is its own fingerprint rather than a field on** :class:`ChunkFingerprint`.
    :attr:`ChunkFingerprint.grammars` is the precedent for folding a library version into the
    chunker's identity, and it does not carry here, for a reason that is structural rather
    than aesthetic:

    - A parser version determines ``text``, which is *upstream* of chunking.
      :class:`ChunkFingerprint` is "the identity of the process that decided where chunks
      begin and end", and a PDF text extractor decided none of that.
    - :class:`ChunkFingerprint` is compared **once per run, for the whole corpus**, and a
      mismatch refuses the run. Parser versions cannot live in a value with that scope and
      still be honest. Recorded only for the parsers that actually ran, the map would grow
      the first time a PDF was ingested and refuse the corpus it had just been part of;
      recorded for every installed parser instead, a ``pypdfium2`` bump would refuse a corpus
      of Markdown that no PDF library has ever touched. ``grammars`` escapes this only
      because the declared language set is configuration, fixed before the run, and read
      without importing anything.
    - One document has exactly one parser. There is no corpus-wide parse identity to compare,
      so the comparison belongs where the fact does: ``documents.parse_fp``, per document.

    So the cost is a third column and a third comparison, and what it buys is invalidation
    that names the documents a bump actually touched.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ("parser", "version", "libraries")

    parser: str = Field(
        min_length=1, description="Registered name of the parser that produced the text."
    )
    version: str = Field(
        min_length=1,
        description="The version of manicule's own extraction rules for that parser — which "
        "blocks it emits, and what its anchors address. Separate from the libraries below "
        "because editing this repository's rules changes stored text just as surely as a "
        "dependency bump does, and would otherwise be the one change nothing records.",
    )
    libraries: dict[str, str] = Field(
        default_factory=dict,
        description="Version by distribution for every library whose behavior decides this "
        "parser's text or anchors, e.g. ``{'pypdfium2': '5.12.1'}``. Keys are PEP 503 "
        "canonical names — ``ruamel-yaml``, never ``ruamel.yaml`` — because the same "
        "distribution spelled two ways is two keys and a bump that appears to change "
        "nothing. Empty is a real and common answer: a parser built on the standard library "
        "alone has no library version to record, and ``version`` above is then the whole of "
        "its identity.",
    )

    @override
    def describe(self) -> str:
        listed = ", ".join(f"{name} {value}" for name, value in sorted(self.libraries.items()))
        return f"{self.parser} rules {self.version} ({listed or 'no parsing libraries'})"


class GlossaryFingerprint(Fingerprint):
    """The identity of the process that decided one document's stored glossary entries.

    Parsing, chunking and embedding were versioned; detection was not, and it is a separate
    stage with rules of its own that change independently of all three. The consequence was an
    index reporting current parser, chunker and embedder fingerprints while serving definitions
    produced by rules that had since been corrected — and reporting itself current while doing
    it, because ``documents.parse_fp`` is what selection reads and no detector change moves it.

    **Detection is downstream of parsing and is not implied by it.** A re-sync of unchanged
    bytes skips before detection runs, so a corrected detector reaches an existing document only
    when something unrelated to the detector happens to that document. Inferring glossary
    freshness from parse freshness would leave the media types nobody bumped stale for ever and
    would migrate the rest by accident, which is coupling that hides the problem rather than
    fixing it.

    **The empty result is a derived result.** A document that states no definitions under the
    current detector records this fingerprint and no entries, and that is a different fact from
    a document whose entries were never computed at all — which records ``NULL``. Only recording
    lineage where there are rows to hang it on would make the two indistinguishable, and would
    look correct on every fixture that happens to contain a definition.
    """

    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ("detector", "rules", "libraries", "middleware")

    detector: str = Field(
        min_length=1,
        description="Which detection strategy produced the entries, or "
        f"``{DETECTION_DISABLED!r}`` when detection was switched off for the run that last "
        "touched this document. A name rather than a boolean beside it, so that the disabled "
        "state is a value somebody reads out of the column rather than a combination of empty "
        "fields they have to interpret.",
    )
    rules: str = Field(
        default="",
        description="A digest over the sources that decide what a definition is — the grammar, "
        "the thresholds, the evidence weights, the boundary model and the normalization that "
        "turns a surface form into a key. Empty when detection is disabled, because none of "
        "them ran.",
    )
    libraries: tuple[str, ...] = Field(
        default=(),
        description="Sorted ``name@version`` for everything outside this repository that "
        "decides a stored entry. A digest of our own sources catches a rule *we* change; it "
        "cannot catch a rule changing underneath an unchanged file, which is what a dependency "
        "upgrade is. Two reach here: ``pydantic``, which validates "
        ":class:`~manicule.core.glossary.GlossaryEntry`'s constraints and so decides which rows "
        "may be persisted at all, and ``unicodedata``, whose character-database version decides "
        "what :func:`~manicule.core.glossary.normalize_acronym` NFKC-folds a surface into — a "
        "stored *lookup key*, moving with the interpreter rather than with any distribution. "
        "Derived from the digested sources' own imports rather than listed, so the one thing "
        "``ParserVersions.distributions`` documents going wrong — a dependency added and not "
        "recorded — cannot happen here. Empty is a real answer for a detector that imports "
        "nothing but the standard library.",
    )
    middleware: tuple[str, ...] = Field(
        default=(),
        description="Sorted ``name@version`` for **every** configured middleware, not only the "
        "ones declaring ``mutates_embedded_text``. That declaration is the wrong filter here "
        "and using it would be a guard that looks right: ``embed_text`` is precisely the field "
        "detection does not read, while the two it does read — a chunk's boundaries, decided by "
        "block metadata a hook may rewrite in ``after_parse``, and its ``heading_path``, which "
        "no digest in "
        ":class:`~manicule.ingest.middleware.MiddlewareRunner` covers — carry no declaration at "
        "all. So the whole chain is named. Empty when detection is disabled.",
    )

    @classmethod
    def disabled(cls) -> GlossaryFingerprint:
        """What a document records when detection was switched off.

        Carries nothing but the state itself. Rules, libraries and middleware are omitted rather
        than recorded-and-ignored, because none of them ran: folding them in would churn the
        lineage of documents nobody detected anything for every time an unrelated hook was
        configured or a dependency moved, and would make two disabled states compare unequal for
        a reason that had no effect.
        """
        return cls(detector=DETECTION_DISABLED)

    @property
    def detects(self) -> bool:
        """Whether this fingerprint describes a detector that ran."""
        return self.detector != DETECTION_DISABLED

    @override
    def describe(self) -> str:
        if not self.detects:
            return "glossary detection disabled"
        listed = ", ".join((*self.libraries, *self.middleware)) or "no libraries, no middleware"
        return f"{self.detector} rules {self.rules} ({listed})"


__all__ = [
    "DETECTION_DISABLED",
    "PROVISIONAL_TOKENIZER_PREFIX",
    "ChunkFingerprint",
    "Fingerprint",
    "GlossaryFingerprint",
    "ParseFingerprint",
]
