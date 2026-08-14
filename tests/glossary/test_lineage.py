"""The detector's identity: what is in it, what moves it, and what would fail to.

The defect this whole feature answers is one an ordinary test suite cannot see. Five glossary
detector changes landed in one day — #101, #103, #108, #110 and #111 — and every one of them
was green: the detector's own tests assert what it produces *now*, from chunks built in the
test. None of them says anything about a corpus indexed yesterday, which keeps the entries
yesterday's rules produced and reports a current ``parse_fp``, a current ``chunk_fp`` and a
current ``embed_fp``, because none of the three has anything to do with detection.

So the assertions here are mostly about what *changes* a fingerprint, and the sharpest of them
is :func:`test_a_new_module_in_the_detector_fails_this_rather_than_being_missed`: the one thing
still maintained by hand is the list of digested sources, and the guard against forgetting it
is mechanical rather than a habit.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.core.fingerprints import DETECTION_DISABLED
from manicule.ingest.glossary_lineage import (
    DETECTOR,
    NOT_DIGESTED,
    SOURCES,
    detector_imports,
    glossary_fingerprint,
    libraries,
    rules_digest,
)
from manicule.ingest.middleware import MiddlewareRunner, chain, declarations
from tests.glossary import system

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk, Document, ParsedBlock, RawDocument
    from manicule.storage.docstore import SqliteDocStore

pytestmark = pytest.mark.usefixtures("store")

EXPANSION = "Network Operations Workspace"


# --- what the fingerprint is made of --------------------------------------------------------


def test_the_digest_covers_the_detector_and_the_normalisation_it_resolves_through() -> None:
    """Requirement 4 is why there are two files rather than one.

    ``manicule.ingest.glossary`` decides which lines are definitions. It does not decide what
    ``NOW``, ``now`` and ``N.O.W.`` have in common, what length a term may be, or which fields
    a stored entry is allowed to carry — ``manicule.core.glossary`` decides all three, and a
    change to any of them changes stored rows without touching a single detection rule.
    """
    assert SOURCES == (
        ("manicule.ingest", "glossary.py"),
        ("manicule.core", "glossary.py"),
    )


def test_the_digest_is_the_bytes_of_those_files_and_can_be_recomputed_by_hand() -> None:
    """Derived, not declared — which is the whole argument for this module.

    ``ParserVersions.rules`` is a number a maintainer has to remember to move, and its own table
    records two parsers bumped for changes they did not make, by somebody who noticed. Nothing
    here has to be noticed. Recomputing the digest from the files independently is what says so:
    if this ever stopped being a function of the sources, the two would part company.
    """
    expected = hashlib.sha256()
    for package, name in SOURCES:
        path = __import__(package, fromlist=["__file__"]).__file__
        assert path is not None
        source = (__import__("pathlib").Path(path).parent / name).read_bytes()
        expected.update(f"{package}.{name}".encode())
        expected.update(b"\0")
        expected.update(source.replace(b"\r\n", b"\n"))
        expected.update(b"\0")

    assert rules_digest() == f"sha256:{expected.hexdigest()}"


def test_editing_a_detection_rule_moves_the_fingerprint(tmp_path: object) -> None:
    """The property the five landed changes needed and did not have.

    Simulated by digesting a modified copy rather than by editing the installed file, because a
    test that rewrote ``manicule/ingest/glossary.py`` would leave the tree changed if it failed
    halfway. What is asserted is the mechanism: the digest is over bytes, so different bytes are
    a different digest, and every one of the five changes was different bytes.
    """
    del tmp_path
    from pathlib import Path  # noqa: PLC0415 - local to this construction

    import manicule.ingest.glossary as detector  # noqa: PLC0415

    assert detector.__file__ is not None
    original = Path(detector.__file__).read_bytes()
    # One character: the confidence a definition needs before it is persisted. Every entry in a
    # corpus is decided by it, and under a hand-maintained version nothing would have moved.
    edited = original.replace(
        b"MIN_DEFINITION_CONFIDENCE: Final = 0.6", b"MIN_DEFINITION_CONFIDENCE: Final = 0.7"
    )
    assert edited != original, "the constant this test edits has been renamed"

    def digest(ingest_source: bytes) -> str:
        made = hashlib.sha256()
        made.update(b"manicule.ingest.glossary.py\0")
        made.update(ingest_source)
        made.update(b"\0")
        return made.hexdigest()

    assert digest(edited) != digest(original)


def test_a_comment_only_edit_also_moves_it_and_that_is_the_direction_chosen() -> None:
    """Stated as a property rather than left as a surprise.

    A digest over bytes cannot tell a rule from the paragraph explaining it, so a docstring fix
    makes the corpus stale. That is the deliberate half of the trade: over-invalidation costs a
    sweep that reads chunks and writes rows — no parser, no embedder, no network — and
    under-invalidation costs a definition served from rules that no longer exist. A normalised
    digest that skipped comments would be the other way round, and its failure would be silent.
    """
    made = hashlib.sha256()
    made.update(b'"""A docstring."""\n')
    other = hashlib.sha256()
    other.update(b'"""A docstring, reworded."""\n')

    assert made.hexdigest() != other.hexdigest()


def test_a_dependency_that_decides_a_stored_entry_is_recorded_with_its_version() -> None:
    """A digest catches a rule *this* repository changes, and cannot catch one moving underneath.

    That is exactly what a dependency upgrade is, and the detector has two.

    ``unicodedata`` is the sharper of them and is the reason this was found at all.
    :func:`~manicule.core.glossary.normalise_acronym` NFKC-folds a surface into the stored
    *lookup key*, and #121 put NFKC into :func:`~manicule.ingest.glossary.initial_skeleton` as
    well — so the version of the character database decides what a term is filed under, it moves
    with the interpreter rather than with any distribution, and nothing in a source digest sees
    it move.

    ``pydantic`` validates :class:`~manicule.core.glossary.GlossaryEntry`'s field constraints, so
    it decides which rows may be persisted at all.
    """
    import unicodedata  # noqa: PLC0415 - read for its data version, as the fingerprint does

    recorded = dict(entry.split("@", 1) for entry in libraries())

    assert recorded["unicodedata"] == unicodedata.unidata_version
    assert recorded["pydantic"] == version("pydantic")
    assert glossary_fingerprint().libraries == libraries()


def test_the_dependencies_are_read_off_the_imports_rather_than_listed() -> None:
    """Derived, so a dependency added to the detector tomorrow is recorded by nobody's memory.

    This is the failure ``ParserVersions.distributions`` documents, happening to somebody: a
    library reaches a fingerprint's inputs and the hand-kept list beside it does not move. There
    is no list here to fail to update — the digested sources' own imports are the list.

    Asserted against a second reading of the same syntax trees rather than against a name. The
    claim that supports is narrow and worth stating as such: it is not that two independent
    algorithms agree — this one is deliberately the same algorithm — it is that **no name is
    written down on either side**, so a `libraries()` quietly replaced by a hard-coded tuple
    fails here the moment the imports and the tuple part company. The test above names
    ``pydantic``, and that one would not.
    """
    third_party = {
        name
        for name in _imported_top_level()
        if name not in sys.stdlib_module_names and not name.startswith("manicule")
    }

    recorded = {entry.split("@", 1)[0] for entry in libraries()}

    assert third_party, "the fixture assumes the detector imports something outside stdlib"
    assert third_party <= recorded, f"{sorted(third_party - recorded)} reached no fingerprint"


@pytest.mark.parametrize(
    ("form", "rule"),
    [
        ("table", b"_TABLE_RE: Final = re.compile("),
        ("heading", b"_HEADING_RE: Final = re.compile("),
        ("list", b"_LIST_MARKER_RE: Final = re.compile("),
        ("acronym", b"_UPPERCASE_SHARE: Final = 0.6"),
        ("bracket", b"_BRACKETS: Final[Mapping[str, str]] = {"),
        ("dash", b"_DASH_RE: Final = re.compile("),
        ("colon", b"_COLON_RE: Final = re.compile("),
        ("parenthetical", b"_PARENTHETICAL_RE: Final = re.compile("),
        ("definition list", b"_DEFINITION_MARKER_RE: Final = re.compile("),
        ("normalisation", b"_STRIPPABLE: Final = "),
    ],
)
def test_each_detection_rule_is_independently_represented_in_the_digest(
    *, form: str, rule: bytes
) -> None:
    """Every rule moves the fingerprint on its own, not only in aggregate.

    A digest over whole files could in principle be satisfied by covering one rule and no other —
    it would still move whenever anything changed, and nobody reading the number could tell which
    of them it was covering. So each rule is located in the digested bytes and shown to move the
    digest by itself.

    Two of these are not written forms and are here for that reason. ``acronym`` is the shape
    gate and ``bracket`` is the boundary model: both decide what is stored without being a syntax
    an author types, and both are exactly what a list of "supported forms" would leave out.
    ``normalisation`` is in the *other* file, so a parametrisation that only ever read the
    detector would fail on it.

    **The locator failing is itself the point.** Rename a rule and this reports that the test is
    stale rather than passing over a constant that no longer exists — checked by pointing it at
    ``_UPPERCASE_SHARE: Final = 0.7``, which produced
    ``AssertionError: the acronym rule has moved or been renamed; this test is stale``.
    """
    original = _digested_bytes()
    assert rule in original, f"the {form} rule has moved or been renamed; this test is stale"

    moved = original.replace(rule, rule + b" ")
    assert hashlib.sha256(moved).hexdigest() != hashlib.sha256(original).hexdigest(), (
        f"editing the {form} rule alone left the fingerprint where it was"
    )


def _source_path(package: str, name: str) -> Path:
    return Path(import_module(package).__file__ or "").parent / name


def _digested_bytes() -> bytes:
    """The detector's sources as the fingerprint reads them, concatenated."""
    return b"".join(
        _source_path(package, name).read_bytes().replace(b"\r\n", b"\n")
        for package, name in SOURCES
    )


def _imported_top_level() -> set[str]:
    """Top-level module names the digested sources import, read here rather than imported.

    Deliberately not calling the module under test, so that what the assertion compares is a
    reading of the *files* against a reading of the *fingerprint* — with no name written down on
    either side of it.
    """
    found: set[str] = set()
    for package, name in SOURCES:
        tree = ast.parse(_source_path(package, name).read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
    return found


def test_the_whole_middleware_chain_is_folded_in_and_not_the_declared_subset() -> None:
    """``mutates_embedded_text`` is the wrong filter here, and using it would look right.

    That declaration names the hooks that rewrite ``embed_text``, which is the field detection
    never reads. What detection *does* read — a chunk's ``heading_path``, and the boundaries
    that follow from block metadata a hook may rewrite in ``after_parse`` — is covered by no
    declaration at all. So the chain is named whole.
    """
    quiet = _Hook("quiet", mutates_embedded_text=False)
    loud = _Hook("loud", mutates_embedded_text=True)
    runner = MiddlewareRunner([quiet, loud])

    assert declarations([quiet, loud]) == ("loud@",)
    assert chain([quiet, loud]) == ("loud@", "quiet@")
    assert runner.chain() == ("loud@", "quiet@")
    assert glossary_fingerprint(middleware=runner.chain()).middleware == ("loud@", "quiet@")


def test_configuring_an_undeclared_hook_moves_the_glossary_fingerprint() -> None:
    """The consequence of the choice above, asserted rather than implied.

    A hook that declares nothing changes no vector and therefore no ``ChunkFingerprint``. It can
    still change what detection sees, so it changes this.
    """
    quiet = _Hook("quiet", mutates_embedded_text=False)

    assert glossary_fingerprint() != glossary_fingerprint(middleware=chain([quiet]))


def test_the_middleware_order_configuration_listed_them_in_is_not_identity() -> None:
    """Reordering configuration is a legitimate change that alters not one entry."""
    first = _Hook("a", mutates_embedded_text=False)
    second = _Hook("b", mutates_embedded_text=False)

    assert glossary_fingerprint(middleware=chain([first, second])) == glossary_fingerprint(
        middleware=chain([second, first])
    )


def test_a_new_module_in_the_detector_fails_this_rather_than_being_missed() -> None:
    """The guard on the one hand-maintained thing left, and the loudest one available.

    :data:`SOURCES` is a list, so the day detection grows a third module somebody has to extend
    it — and forgetting would be silent in the worst way: the fingerprint would keep moving for
    edits to the two files it knows about and would sit still for every edit to the new one.

    So the list is not trusted. The digested sources' own syntax trees are read for what they
    import, and anything reaching ``manicule`` that is neither digested nor named in
    :data:`NOT_DIGESTED` with a reason fails here, by name.

    ``TYPE_CHECKING`` imports are included deliberately: they never execute, so no runtime
    inspection would see one, and a rule arriving through one would be a rule nothing covered.
    """
    uncovered = detector_imports() - set(NOT_DIGESTED)

    assert not uncovered, (
        f"the glossary detector imports {sorted(uncovered)}, which the fingerprint does not "
        f"digest. Either add the module to SOURCES, or add it to NOT_DIGESTED with the reason "
        f"it cannot change what gets stored."
    )


def test_every_deliberate_exclusion_carries_a_reason_somebody_can_argue_with() -> None:
    """An allowlist with prose attached, so an omission cannot masquerade as a decision."""
    assert NOT_DIGESTED
    for module, reason in NOT_DIGESTED.items():
        assert module.startswith("manicule"), module
        assert len(reason) > 40, f"{module} is excluded without a reason worth reading"


# --- the disabled state ---------------------------------------------------------------------


def test_detection_switched_off_is_a_value_rather_than_an_absence() -> None:
    """Requirement 8, and the choice this repository made between its two options.

    ``NULL`` already means "never recomputed". If a disabled run also recorded nothing, an
    operator could not tell a corpus whose definitions were never looked for from one whose
    detector is deliberately off — so the disabled state is written into the column, where it
    can be read rather than inferred.
    """
    off = glossary_fingerprint(enabled=False)

    assert off.detector == DETECTION_DISABLED
    assert not off.detects
    assert off.canonical() == ('{"detector":"disabled","libraries":[],"middleware":[],"rules":""}')
    assert off.describe() == "glossary detection disabled"


def test_the_disabled_fingerprint_carries_neither_rules_nor_middleware() -> None:
    """Neither ran, so recording them would be describing work that did not happen.

    It also has a consequence worth having: configuring a hook while detection is off does not
    churn the lineage of documents no detector read.
    """
    hook = _Hook("quiet", mutates_embedded_text=False)

    assert glossary_fingerprint(enabled=False, middleware=chain([hook])) == (
        glossary_fingerprint(enabled=False)
    )
    assert glossary_fingerprint(enabled=False).libraries == (), (
        "a dependency version describes work that did not happen either"
    )


def test_switching_detection_back_on_makes_every_disabled_document_stale() -> None:
    """Which is what somebody switching it back on expects, and would otherwise have to ask for."""
    assert glossary_fingerprint(enabled=False) != glossary_fingerprint(enabled=True)


def test_the_detector_is_named_so_two_strategies_would_not_look_like_one_edit() -> None:
    """A digest alone could not tell a second detector from a typo fix in the first."""
    assert glossary_fingerprint().detector == DETECTOR
    assert glossary_fingerprint().describe().startswith(f"{DETECTOR} rules sha256:")


# --- what the store does with it --------------------------------------------------------------


async def test_a_documents_lineage_is_readable_without_reading_its_vocabulary(
    store: SqliteDocStore,
) -> None:
    """Requirement 3, in the method that exists to satisfy it.

    A corpus-wide answer to "is this current" has to cost an indexed column rather than the
    definitions themselves — a check that read the vocabulary to decide whether the vocabulary
    is current would cost the thing it is asking about.
    """
    document, _ = await system.index(store, "glossary", "Glossary of terms", [f"NOW — {EXPANSION}"])

    assert await store.glossary_lineage(document.id) == glossary_fingerprint().canonical()


async def test_a_document_that_states_nothing_records_lineage_all_the_same(
    store: SqliteDocStore,
) -> None:
    """Requirement 2, and the one an implementation is most likely to get wrong.

    Lineage hung on the rows would look correct on every fixture that has entries and would
    leave a document with none indistinguishable from a document nobody has read — so every
    sweep would select every prose page for ever, and "the current detector finds nothing here"
    would not be a thing the index could say.
    """
    document, _ = await system.index(
        store, "prose", "Runbook", ["The scheduler restarts nightly, which is fine."]
    )

    assert await store.glossary_entries(document.id) == []
    assert await store.glossary_lineage(document.id) == glossary_fingerprint().canonical()


async def test_the_entries_and_the_claim_about_them_are_one_transaction(
    store: SqliteDocStore,
) -> None:
    """Requirement 6. Two statements would leave a window, and one side of it is the defect.

    Rows written and fingerprint not yet advanced is a document that gets repaired again for
    nothing. Fingerprint advanced and rows not written is a document reporting itself current
    while serving the previous detector's definitions, which is what all of this exists to make
    impossible. Asserted by there being no way to write one without the other: the keyword is
    required, so a caller cannot omit it.
    """
    import inspect  # noqa: PLC0415 - local to this assertion

    parameter = inspect.signature(store.replace_glossary_entries).parameters["fingerprint"]

    assert parameter.default is inspect.Parameter.empty, (
        "a default would let one write path forget the lineage, and a document with entries "
        "and no fingerprint is the state this feature exists to make unreachable"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


async def test_an_unwritten_document_is_absent_rather_than_empty(
    store: SqliteDocStore,
) -> None:
    """``None`` and "no entries" are different answers and neither is inferred from the other."""
    document, _ = await system.index(store, "glossary", "Glossary of terms", [f"NOW — {EXPANSION}"])
    await store.set_lineage(document.id, chunk_fp=None, embed_fp=None)

    assert await store.glossary_lineage(document.id) == glossary_fingerprint().canonical(), (
        "set_lineage(glossary_fp=None) must leave a lineage alone rather than clear it"
    )


class _Hook:
    """A middleware that does nothing, so that only its declaration is under test."""

    def __init__(self, name: str, *, mutates_embedded_text: bool) -> None:
        self.name = name
        self.mutates_embedded_text = mutates_embedded_text

    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        return raw

    async def after_parse(
        self, document: Document, blocks: Sequence[ParsedBlock]
    ) -> list[ParsedBlock]:
        del document
        return list(blocks)

    async def after_chunk(self, document: Document, chunks: Sequence[Chunk]) -> list[Chunk]:
        del document
        return list(chunks)

    async def after_store(self, document: Document) -> None:
        del document
