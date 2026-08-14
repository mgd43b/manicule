"""What the glossary detector was built out of, so a change to it can name the documents it moved.

:mod:`manicule.parsers.versions` is the same module one stage upstream, and the difference
between them is the whole design of this one.

**A parser's identity is mostly other people's code, so it is looked up. A detector's identity
is entirely ours, so it is derived.** ``ParserVersions.rules`` is a hand-maintained number
beside a list of distributions, and it has to be, because manicule's own extraction rules are
the one input no dependency map records. That table's comments are candid about the cost: two
parsers there carry a bump *for a change they did not themselves make*, written down by
somebody who noticed. A number a person has to remember to move is a number that will one day
not be moved, and the failure is silent — a corpus that reports itself current.

Detection has no dependencies at all. ``docs/retrieval.md`` §14 requires it to be
deterministic and model-free, so the modules below are pure Python over regular expressions,
and **every input to what gets stored is a byte in one of two files**. That makes the honest
identity a digest of those files rather than a version somebody maintains, and the five
detector changes that motivated this module — a list-marker rule, a widened initials matcher, a
narrowed page-context rule, a form-and-evidence gate on headings, a bracket-aware boundary
model — would each have moved it with nobody to remember anything.

**The digest is deliberately too sensitive rather than too lenient**, and the trade is
affordable in a way it would not be one stage up. A comment-only edit to either file moves it
and makes the corpus stale. What that costs is a sweep that reads stored chunks, runs regular
expressions over them and writes rows: no connector, no parser, no embedder, no GPU and no
network — the cheapest repair in the system. The alternative is a normalised digest that skips
comments and docstrings, and its failure mode is the one this exists to remove: a normalisation
with a gap in it drops a real change silently, and silence is the defect. Over-invalidation
announces itself in a sweep an operator can watch finish.

**What the digest does not cover, and how forgetting is caught.** :data:`SOURCES` is a list, and
a list is a thing somebody can fail to extend the day detection grows a third module. So it is
not trusted: :func:`detector_imports` reads what those files actually import out of their own
syntax trees, and ``tests/glossary/test_lineage.py`` fails unless every ``manicule`` module they
reach is either digested or named in :data:`NOT_DIGESTED` with a reason. That is the loudest
guard available against the only hand-maintained thing left here, and it is mechanical rather
than a habit.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Final

from manicule.core.fingerprints import GlossaryFingerprint

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DETECTOR",
    "NOT_DIGESTED",
    "SOURCES",
    "detector_imports",
    "glossary_fingerprint",
    "libraries",
    "rules_digest",
]

DETECTOR: Final = "deterministic"
"""The name of the only detection strategy manicule ships.

Recorded rather than assumed, and it is the field that would tell two strategies apart if a
second ever arrived — the same job :attr:`~manicule.core.fingerprints.ChunkFingerprint.chunker`
does for chunking. A digest alone could not: two detectors' sources hash differently, but so
does one detector's source after a typo fix, and a reader of the stored column would have no
way to tell which kind of change they were looking at.
"""

SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("manicule.ingest", "glossary.py"),
    ("manicule.core", "glossary.py"),
)
"""Every file whose bytes decide which entries a document stores, as package and file name.

``manicule.ingest.glossary``
    The detector. Written forms, evidence weights, :data:`the persistence threshold
    <manicule.ingest.glossary.MIN_DEFINITION_CONFIDENCE>`, the description-boundary model, the
    list-marker rule, the heading gate and the initials matcher — the set *and* the contents of
    what is stored, all of it.

``manicule.core.glossary``
    Normalisation, and it is in here for requirement 4 rather than by association.
    :func:`~manicule.core.glossary.normalise_acronym` decides the lookup key,
    ``MIN_ACRONYM_LENGTH`` and ``MAX_ACRONYM_LENGTH`` decide which surfaces can be terms at
    all, :class:`~manicule.core.glossary.DefinitionForm` names the forms whose weights the
    detector reads, and :class:`~manicule.core.glossary.GlossaryEntry`'s field constraints
    decide what may be persisted. A change to any of them changes stored rows without touching
    the detector.

Read as bytes, in this order, with CRLF folded to LF so that two checkouts of one commit agree.
Line endings are the one difference between installations that says nothing about behaviour,
and leaving it in would make a corpus restored onto a differently-configured machine stale for
a reason nobody could act on.
"""

NOT_DIGESTED: Final[dict[str, str]] = {
    "manicule.core.content": (
        "the chunks handed to detection are its input, not its rules. Their identity is "
        "ChunkFingerprint's, and nothing can change them without rewriting `chunks` — which "
        "runs detection again through the ordinary ingest path, so there is no way for this "
        "module to move behind a stored glossary's back."
    ),
}
"""``manicule`` modules the digested sources import that are deliberately left out, with why.

An allowlist with reasons attached, checked by a test, rather than a silent filter. Every entry
here is a judgement that something reachable from the detector cannot change what it stores,
and a judgement stated in prose can be argued with; an omission cannot.

One more exclusion is worth stating even though it is not reachable from these files and so
never reaches this map. :func:`~manicule.core.ids.glossary_entry_id` decides a stored row's
primary key, and it is imported by the *store* rather than by the detector. A change to it would
rewrite every entry id — but an id is derived from the entry's own content, resolves nothing
outside the row, and is rewritten together with its aliases inside one transaction, so it cannot
produce a wrong answer to any question a reader asks. Digesting the whole of ``core/ids.py``
to cover it would make a change to *document* identity advance glossary lineage, which is a
rule far wider than its justification.
"""


def _read(package: str, name: str) -> bytes:
    """One digested source's bytes, newline-folded.

    Read through :func:`importlib.resources.files` rather than off ``__file__`` so that an
    installation serving manicule out of a zip answers the same way a checkout does.

    Raises:
        FileNotFoundError: The source is not on this installation. Nothing is caught and no
            placeholder is substituted, for the reason
            :func:`~manicule.parsers.versions.parse_fingerprint` gives about a missing
            distribution: a constant standing in for a thing that varies is a fingerprint that
            never moves, which is worse than having none.
    """
    data = files(package).joinpath(name).read_bytes()
    return data.replace(b"\r\n", b"\n")


@cache
def rules_digest() -> str:
    """A digest over every source in :data:`SOURCES`, as ``sha256:<hex>``.

    Names are hashed alongside their contents and separated by NUL, for the reason
    :func:`~manicule.ingest.middleware.text_digest` gives: concatenating the files alone would
    let a line moved from the end of one to the start of the next hash identically, which is the
    difference between a check and the appearance of one.

    Cached for the life of the process. It is a fact about the files on disk, which do not
    change under a running interpreter, and ingest asks it once per document.
    """
    digest = hashlib.sha256()
    for package, name in SOURCES:
        digest.update(f"{package}.{name}".encode())
        digest.update(b"\0")
        digest.update(_read(package, name))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@cache
def libraries() -> tuple[str, ...]:
    """``name@version`` for everything outside this repository that decides a stored entry.

    **Derived from the sources' own imports, not from a list somebody maintains**, which is the
    same argument as :func:`rules_digest` one level out. A digest catches a rule *this* repository
    changes; it cannot catch a rule changing underneath an unchanged file, and that is exactly
    what a dependency upgrade is.

    Two kinds are recorded and neither was, until the specification asked:

    **Distributions the digested sources import.** ``manicule.core.glossary`` imports
    ``pydantic``, which validates :class:`~manicule.core.glossary.GlossaryEntry`'s field
    constraints and therefore decides which rows may be persisted at all. Found by reading the
    imports rather than by naming it, so a second one added to the detector tomorrow is recorded
    without anybody remembering — the failure mode ``ParserVersions.distributions`` documents.

    **The Unicode database.** :func:`~manicule.core.glossary.normalise_acronym` NFKC-normalises,
    so the version of the character database decides a stored *lookup key*; #121 put NFKC into
    :func:`~manicule.ingest.glossary.initial_skeleton` as well. It moves with the interpreter and
    has no distribution to look up, so it is the one input the derivation above cannot see and
    the one entry here that is named. Recorded as ``unicodedata@<unidata_version>`` so that it
    reads like the rest.

    Sorted, so the value is a set of facts rather than an import order.

    Raises:
        PackageNotFoundError: A distribution the detector imports is not installed, which means
            the detector could not have run. Answering with a partial set would be a fingerprint
            that claims fewer inputs than it has.
    """
    import unicodedata  # noqa: PLC0415 - see above; read for its data version, not its functions
    from importlib.metadata import packages_distributions, version  # noqa: PLC0415

    installed = packages_distributions()
    named = {
        distribution
        for module in _third_party_imports()
        for distribution in installed.get(module, [module])
    }
    return tuple(
        sorted(
            [f"unicodedata@{unicodedata.unidata_version}"]
            + [f"{name}@{version(name)}" for name in named]
        )
    )


def glossary_fingerprint(
    *, enabled: bool = True, middleware: Sequence[str] = ()
) -> GlossaryFingerprint:
    """What detection would record for a document ingested now.

    Args:
        enabled: Whether ``rag.glossary.detect_on_ingest`` is on. When it is not, the answer is
            :meth:`~manicule.core.fingerprints.GlossaryFingerprint.disabled` and none of the
            other three is read — they describe work that did not happen.
        middleware: ``name@version`` for every configured hook, from
            :meth:`~manicule.ingest.middleware.MiddlewareRunner.chain`. Sorted and
            de-duplicated here rather than at the call site, so the identity does not depend on
            the order configuration happened to list them in.
    """
    if not enabled:
        return GlossaryFingerprint.disabled()
    return GlossaryFingerprint(
        detector=DETECTOR,
        rules=rules_digest(),
        libraries=libraries(),
        middleware=tuple(sorted(set(middleware))),
    )


def detector_imports() -> frozenset[str]:
    """Every ``manicule`` module the digested sources import, read out of their syntax trees.

    The input to the guard described in this module's docstring, and it reads the files rather
    than the loaded modules on purpose: ``import`` statements under ``TYPE_CHECKING`` never
    execute, so an imported-for-typing module is invisible to any runtime inspection and
    perfectly visible here. A rule that arrived through one of those would otherwise be a rule
    the digest did not cover and nothing reported.

    Relative imports are resolved against the source's own package, and submodule imports are
    reported at the module that was named — ``from manicule.core.glossary import X`` is
    ``manicule.core.glossary``, not ``manicule.core``.
    """
    found = {name for name in _imported_modules() if name.startswith("manicule")}
    digested = {f"{package}.{name.removesuffix('.py')}" for package, name in SOURCES}
    return frozenset(found - digested)


def _imported_modules() -> frozenset[str]:
    """Every module the digested sources import, at full dotted name.

    One scan, two readers: :func:`detector_imports` keeps the ``manicule`` ones and asks whether
    the digest covers them, and :func:`_third_party_imports` keeps the rest and asks what version
    they are. Written once because the two questions are the same question about different halves
    of one import list, and a second walk would be a second chance to read a syntax tree
    differently.
    """
    found: set[str] = set()
    for package, name in SOURCES:
        tree = ast.parse(_read(package, name))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # `level` is the number of leading dots; zero is an absolute import.
                if node.level:
                    found.add(f"{package}.{node.module}" if node.module else package)
                elif node.module:
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
    return frozenset(found)


def _third_party_imports() -> frozenset[str]:
    """Top-level modules the digested sources import from outside the standard library.

    ``sys.stdlib_module_names`` is the interpreter's own answer to "is this the standard
    library", which is the right authority: a hand-kept exclusion list would be one more thing to
    forget, and forgetting one here means recording a version for something that has none rather
    than missing one that matters. ``__future__`` is in that set, as are ``re`` and
    ``unicodedata`` — the last of which is why :func:`libraries` records a data version by hand.
    """
    return frozenset(
        name.split(".")[0]
        for name in _imported_modules()
        if not name.startswith("manicule") and name.split(".")[0] not in sys.stdlib_module_names
    )
