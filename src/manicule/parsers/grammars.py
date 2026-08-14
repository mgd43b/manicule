"""The declared language set, its grammars, and the refusal when one is absent.

``tree-sitter-language-pack`` is MIT and its manifest lists 371 languages, but **the
grammars are not in the wheel**. They are fetched from a release bundle on first use, into a
per-user cache. A fresh install therefore has zero grammars and needs network egress the
first time it is asked to parse a ``.py`` file.

Left alone that is a corpus-consistency hazard: code would chunk one way on a machine that
reached the network and another way on a machine that did not — different boundaries,
different embeddings, one corpus. Platform may change *throughput*; it must never change
*output*. This module closes it in three moves, and all three are required:

1. **A declared language set** (:data:`DECLARED_LANGUAGES`), pinned in configuration rather
   than discovered from whatever happens to be cached. manicule supports the languages it
   declares and no others. Keys are checked against the manifest by
   :func:`validate_languages` at construction time, not at first use, so a typo is a
   configuration error rather than a document that mysteriously will not parse.
2. **Pre-seed, never lazy-load.** :func:`prefetch` is what ``manicule init`` and
   ``manicule doctor --fix`` call. The cache directory and language set are fixed through
   the pack's own configuration entry point, and the manifest URL is overridable so an
   air-gapped deployment can point at an internal mirror. Before any of that it seeds from an
   **offline bundle** if one is installed (:mod:`manicule.parsers.grammar_bundle`), which is
   what makes a pre-seed succeed on a host with no route to anything at all.

   Those two command names are a claim, and this module used to make it while neither command
   called anything here — the pre-seed had no caller at all and ``--fix`` did not exist.
   ``tests/parsers/test_grammars.py`` now runs every command named anywhere in this file and
   fails if it is absent, if the flag it is quoted with is absent, or if running it does not
   reach :func:`prefetch`. A docstring that names its caller is checkable, so it is checked.
3. **A missing grammar is a refusal, not a fallback.** :func:`load_parser` raises
   :class:`GrammarUnavailableError`; there is no line-splitting fallback, because a silent
   fallback is precisely how two machines end up with two chunkings of one file. A grammar
   that is *present and will not load* refuses through the same door
   (:class:`GrammarUnusableError`), because the pack reports it as downloaded and then fails
   with "language not found", which reads as a routing mistake rather than a broken file.

**The pack API this module is built on, and why each call was chosen.** Established by
reading the installed package rather than from documentation, because the obvious call is
the wrong one twice over:

``SupportedLanguage``
    The 371 names, as a ``Literal`` in the wheel. This is what a declared key is validated
    against, and the choice is not the obvious one. ``manifest_languages()`` reads like the
    right call and **is not offline**: it reads a manifest file from the cache and fetches
    that file when the cache lacks it, so on a fresh install it raises a download error
    instead of answering — which would make validating the declared set require the network
    that the declared set exists to stop depending on. ``has_language()`` does answer
    offline, but it accepts *aliases*, which would let through a name that
    ``downloaded_languages()`` never reports. The ``Literal`` is plain data in the wheel and
    answers with no I/O at all.

``downloaded_languages()``
    The languages whose shared library is already in the cache. **This is the absence check,
    and it is the only one that does not touch the network.** ``available_languages()``
    reports the same set under a name that reads like the manifest; using it would work but
    invites the reader to think it means "supported".

``get_parser(name)``
    Loads a grammar — **and downloads it if the cache lacks it.** So it is never called
    without :func:`is_available` having answered first; that ordering is what makes the
    refusal a refusal instead of an implicit fetch.

``get_tags_query(name)``
    The pack's own tags query for a language, as source text, or ``None``. There are no
    ``.scm`` files in the wheel — the patterns are compiled into the same native library
    that contains no grammars, which is exactly why they survive an offline install when
    the grammars do not. 71 of the 371 languages have one.

``configure(PackConfig(...))`` / ``prefetch([...])``
    Apply cache and language configuration without downloading, and download-then-load.
    Both act on the pack's single process-global registry, which is why
    :func:`configure_pack` is called once when a parser is built rather than per document.

The manifest URL has no field on ``PackConfig``; the only hook the native library reads is
the environment variable :data:`MANIFEST_URL_ENV`, confirmed by symbol inspection and by
observing that it is read lazily at fetch time rather than at import.
"""

from __future__ import annotations

import difflib
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, get_args

from manicule.core.errors import ConfigError, ManiculeError

if TYPE_CHECKING:
    from tree_sitter import Parser, Query

__all__ = [
    "DECLARED_LANGUAGES",
    "DEFAULT_SCOPE_SEPARATOR",
    "DEFINITION_WRAPPERS",
    "FETCH_RETRY_DELAYS",
    "MANIFEST_URL_ENV",
    "MEDIA_TYPES",
    "NODE_TYPE_DEFINITIONS",
    "PACK_DISTRIBUTION",
    "PRESEED_COMMAND",
    "SCOPE_SEPARATORS",
    "DefinitionRule",
    "GrammarFetchError",
    "GrammarUnavailableError",
    "GrammarUnusableError",
    "bundle_status",
    "cache_directory",
    "configure_pack",
    "grammar_versions",
    "is_available",
    "language_for_media_type",
    "load_parser",
    "missing_grammars",
    "pack_version",
    "prefetch",
    "scope_separator",
    "tags_query",
    "tags_query_source",
    "validate_languages",
]

_supported: frozenset[str] | None = None
_default_cache_dir: str | None = None

MANIFEST_URL_ENV: Final = "TREE_SITTER_LANGUAGE_PACK_MANIFEST_URL"
"""Environment variable the pack reads to locate the grammar manifest.

The mirror hook for an air-gapped deployment. It is an environment variable rather than a
configuration field because that is the only override the native library exposes; setting it
from :func:`configure_pack` keeps that fact in one place instead of in every deployment's
shell profile.
"""

PACK_DISTRIBUTION: Final = "tree-sitter-language-pack"
"""The installed distribution the grammars come from."""

FETCH_RETRY_DELAYS: Final[tuple[float, ...]] = (1.0, 4.0)
"""Seconds to wait between download attempts. Its length is how many retries follow the first.

The grammars come from a third-party GitHub release, and the failure that host actually
produces is not a refusal — it is ``Peer disconnected`` part-way through a transfer, which the
next attempt completes. Without a retry that single dropped connection fails whatever asked for
it; in this repository that was every pull request at once, because the image build pre-seeds.

**This tolerates a flaky transfer and not an absent host, and the difference is kept visible.**
The delays are bounded and few, the last failure is raised with everything the first one would
have said, and no attempt anywhere returns a partial set of grammars — :func:`prefetch` asks the
cache afterwards regardless of what the pack reported. A retry that quietly shipped what it
managed to get would be strictly worse than the outage it was added for.

Backoff rather than a fixed interval because the two transient failures differ: a dropped
connection clears at once, a rate limit does not, and two attempts a second apart would spend
both inside the same rate-limit window.

Read at call time rather than captured, so a caller that must not sleep — the test suite, which
would otherwise spend five seconds per deliberately-unreachable manifest — can say so.
"""

PRESEED_COMMAND: Final = "manicule doctor --fix"
"""The command that seeds a missing grammar, named once because it is quoted everywhere.

It is the ``status_detail`` of every document refused for want of a grammar, so it is the one
string an operator is most likely to read about this subsystem — and for a while it named a
flag that did not exist. Naming it here means the refusal, the diagnostic and the test that
runs the command all say the same thing, and a rename cannot leave two of the three behind.
"""


def pack_version() -> str:
    """The installed grammar pack release. See :func:`grammar_versions`.

    Read from the distribution's metadata rather than from ``pack.__version__``, which is the
    same string — the suite asserts that rather than assuming it. Reading the metadata answers
    without importing the pack, so recording a grammar version costs no native extension load
    on a machine whose corpus contains no code. It also answers on a machine where the native
    extension cannot be loaded at all, which is the honest behavior: a version is a fact about
    what is installed, not about what has been initialized.
    """
    from importlib.metadata import version  # noqa: PLC0415 - see docstring

    return version(PACK_DISTRIBUTION)


MEDIA_TYPES: Final[Mapping[str, str]] = {
    "bash": "text/x-shellscript",
    "c": "text/x-c",
    "cpp": "text/x-c++",
    "csharp": "text/x-csharp",
    "css": "text/css",
    "dart": "text/x-dart",
    "elixir": "text/x-elixir",
    "go": "text/x-go",
    "java": "text/x-java",
    "javascript": "text/javascript",
    "kotlin": "text/x-kotlin",
    "lua": "text/x-lua",
    "ocaml": "text/x-ocaml",
    "php": "text/x-php",
    "python": "text/x-python",
    "r": "text/x-r",
    "ruby": "text/x-ruby",
    "rust": "text/x-rust",
    "scala": "text/x-scala",
    "sql": "application/sql",
    "swift": "text/x-swift",
    "tsx": "text/x-tsx",
    "typescript": "text/x-typescript",
    "zig": "text/x-zig",
}
"""Media type per declared language. The routing table, and the reason the set is declared.

``text/javascript``, ``text/css`` and ``application/sql`` are IANA-registered; everything
else uses the conventional ``text/x-`` form, since most programming languages have no
registration. One language, one media type, and no two languages share one — asserted by
this module's tests, because a collision would make routing depend on dictionary order.

Notably absent, and deliberately: HTML, JSON, YAML, TOML and Markdown all have grammars in
the pack, and all are routed elsewhere (``docs/parsing.md`` §2.4). Claiming ``text/html``
here would put two parsers on one media type, and the winner would be whichever registered
last.
"""

DECLARED_LANGUAGES: Final[tuple[str, ...]] = tuple(sorted(MEDIA_TYPES))
"""The languages manicule supports. Sorted, so the set is one value rather than an order.

Derived from :data:`MEDIA_TYPES` rather than listed twice: a language with no media type
routes no documents, and a media type with no language routes documents nowhere.
"""

_LANGUAGES_BY_MEDIA_TYPE: Final[Mapping[str, str]] = {
    media_type: language for language, media_type in MEDIA_TYPES.items()
}


DEFAULT_SCOPE_SEPARATOR: Final = "."

SCOPE_SEPARATORS: Final[Mapping[str, str]] = {
    "cpp": "::",
    "php": "::",
    "ruby": "::",
    "rust": "::",
}
"""How a language writes a qualified name, where it is not ``.``.

``symbol`` is read by people who know the language, and ``Anchor.render`` for Rust reads as
a mistake. Ruby is here alongside the three ``docs/parsing.md`` §8.2 names because ``::`` is
how Ruby itself writes a nested constant; every other declared language writes ``.``.
"""


@dataclass(frozen=True, slots=True)
class DefinitionRule:
    """Where a definition node of a given type keeps its name.

    Two shapes because grammars use both. ``field`` is the tidy case — the grammar labels
    the name node, and ``child_by_field_name`` finds it. ``child_type`` is for grammars that
    label nothing and simply place an identifier among the children; the first child of that
    type is the name, which holds because a definition's own name precedes its body.
    """

    field: str | None = None
    child_type: str | None = None


_ECMA_DEFINITIONS: Final[Mapping[str, DefinitionRule]] = {
    "class_declaration": DefinitionRule(field="name"),
    "function_declaration": DefinitionRule(field="name"),
    "method_definition": DefinitionRule(field="name"),
}

NODE_TYPE_DEFINITIONS: Final[Mapping[str, Mapping[str, DefinitionRule]]] = {
    "bash": {"function_definition": DefinitionRule(field="name")},
    "rust": {"impl_item": DefinitionRule(field="type")},
    "tsx": _ECMA_DEFINITIONS,
    "typescript": _ECMA_DEFINITIONS,
    "zig": {"FnProto": DefinitionRule(child_type="IDENTIFIER")},
}
"""Definition node types this repository names itself, consulted per node after the tags
query has been asked.

Per node rather than per language, because the gap is per node. Three of these entries exist
for a reason established by running the queries rather than by reading about them:

- **Rust** — the pack's tags query names ``struct_item``, ``function_item`` and
  ``mod_item``, but not ``impl_item``, so a method would come back as ``total`` instead of
  ``Store::total``. ``impl_item`` keeps its type under the ``type`` field, not ``name``,
  which is why :class:`DefinitionRule` addresses a field by name.
- **TypeScript and TSX** — the exposed query is the TypeScript-specific half of the upstream
  pair and covers signatures, interfaces, abstract classes and modules; the ordinary
  ``class_declaration``, ``function_declaration`` and ``method_definition`` patterns live in
  the ECMAScript half, which the pack does not expose separately. Without these three
  entries every ordinary TypeScript class and function is unnamed. JavaScript is unaffected
  — its query is complete — and the entries are harmless where the query already answers,
  since the query is asked first.
- **Zig** — its ``FnProto`` node labels no field at all and carries a bare ``IDENTIFIER``
  child, which is the second shape :class:`DefinitionRule` supports. Only functions are
  listed: mapping ``VarDecl`` too would name every ``const`` in the file, and a symbol that
  names an import is worse than no symbol.

``css`` and ``sql`` are declared, have no tags query and no entry here, so their blocks carry
``symbol=None`` — a :class:`~manicule.core.anchors.LineAnchor` with exact lines and no
symbol, which is honest and still cites correctly. Because the pack version is pinned and the
language set is declared, which languages fall into that case is identical on every machine.
"""

DEFINITION_WRAPPERS: Final[Mapping[str, frozenset[str]]] = {
    "javascript": frozenset({"export_statement"}),
    "python": frozenset({"decorated_definition"}),
    "tsx": frozenset({"export_statement"}),
    "typescript": frozenset({"export_statement"}),
}
"""Node types that wrap a definition without being one.

``@cache`` above a class and ``export`` in front of one both produce a node that contains the
definition and is not itself named, so a block covering the decorator *and* the class is
contained by the wrapper and by nothing narrower. Unwrapping it is the difference between
``TokenStore.refresh`` and no symbol at all for every decorated or exported definition.

Listed explicitly per language rather than detected by shape. The shape test — "if the
containing node is unnamed, look at its last child" — also fires on a block that spans a
whole file or a whole function body, and names it after whatever happens to come last, which
is a wrong symbol rather than a missing one. Most languages need no entry: Rust attributes,
Java annotations and C# attributes all sit *inside* the declaration node.
"""


class GrammarUnavailableError(ManiculeError):
    """A declared language's grammar is not in the cache, so the document is refused.

    Deliberately **not** a :class:`~manicule.core.errors.ParseError`. A ``ParseError`` means
    "not my kind of document" and hands the document to the next parser in the fallback
    chain — which, for source code, is the plain-text parser, and that would line-split the
    file. Line-splitting is the exact outcome the declared language set exists to prevent,
    so this error stops the document instead of passing it on: it is stored, visible, and
    re-indexable the moment the grammar arrives.
    """

    def __init__(
        self, language: str, reason: str | None = None, message: str | None = None
    ) -> None:
        self.language = language
        self.reason = reason or f"grammar unavailable: {language} — run {PRESEED_COMMAND}"
        """The document's ``status_detail``. Names the language and the command that fixes
        it, because "unsupported" on its own tells an operator nothing to do."""
        super().__init__(
            message
            or (
                f"{self.reason}. The grammar for {language!r} was not found in the cache at "
                f"{cache_directory()}. It is not fetched on demand on purpose: a file that "
                f"chunks one way here and another way on a machine that reached the network "
                f"produces one corpus with two chunkings."
            )
        )


class GrammarUnusableError(GrammarUnavailableError):
    """A declared language's grammar is in the cache and the pack cannot load it.

    A distinct failure from an absent grammar, and one that is invisible without this class.
    ``downloaded_languages()`` reports a language whose library file exists, whatever that file
    contains — so a truncated download, a library built for another platform, or a bundle
    copied from an x86 machine to an Apple Silicon one all report as *present* and then fail at
    ``get_parser`` with ``Language 'python' not found``. That message reads as a routing
    mistake, and left unhandled it reaches the parser chain as an ordinary exception, which
    advances the chain — so the document is offered to whatever comes next.

    A subclass rather than a sibling, so that everything already written to stop on a missing
    grammar stops on an unusable one too. Only the ``reason`` differs, because only the remedy
    does: the fix is to replace the file, not to fetch one that is already there.
    """

    def __init__(self, language: str, detail: str) -> None:
        reason = f"grammar unusable: {language} — run {PRESEED_COMMAND}"
        super().__init__(
            language,
            reason=reason,
            message=(
                f"{reason}. The grammar for {language!r} is present in the cache at "
                f"{cache_directory()} and the grammar pack could not load it: {detail}. A "
                f"library built for another platform or truncated in transit looks exactly "
                f"like this. Re-seed it from an offline bundle built on this platform, or "
                f"delete it and pre-seed again."
            ),
        )


class GrammarFetchError(ManiculeError):
    """Grammars could not be fetched, so the declared set cannot be completed.

    Separate from :class:`GrammarUnavailableError` because the remedy is different: that one
    says "run the command", this one is raised *by* that command and says which languages,
    which manifest URL, and what the offline bundle had to offer — so an air-gapped deployment
    can see whether it is pointing at the public manifest rather than at its mirror, and
    whether it has a bundle at all.
    """


def validate_languages(languages: Sequence[str]) -> tuple[str, ...]:
    """Check declared language keys against the pack manifest, and canonicalize the set.

    Runs when a parser is constructed rather than when a document arrives, so that a typo is
    a startup failure with the alternatives listed rather than a document that quietly never
    parses.

    Args:
        languages: The declared keys, in any order.

    Returns:
        The same keys, deduplicated and sorted, so that the declared set is a value rather
        than an order — two configurations listing the same languages differently describe
        the same corpus.

    Raises:
        ConfigError: The set is empty, a key is not in the manifest, or a key is a real
            grammar manicule does not declare a media type for. The first message lists the
            closest manifest names, which is what turns the ``csharp``/``c_sharp`` trap from
            a lookup failure into a one-line fix; the second is a different mistake with a
            different fix, so it is a different message.
    """
    if not languages:
        msg = (
            "the code parser was declared with no languages, so no document would ever "
            f"route to it. Declare a subset of {len(DECLARED_LANGUAGES)} supported keys, "
            f"e.g. {list(DECLARED_LANGUAGES[:4])}."
        )
        raise ConfigError(msg)

    known = _supported_names()
    unknown = sorted({language for language in languages if language not in known})
    if unknown:
        hints = {language: difflib.get_close_matches(language, known, n=3) for language in unknown}
        detail = "; ".join(
            f"{language!r} (did you mean {matches}?)" if matches else repr(language)
            for language, matches in hints.items()
        )
        msg = (
            f"unknown grammar language key(s): {detail}. Keys are the grammar pack's own "
            f"names, of which there are {len(known)}; the one that catches people is C#, "
            f"which is 'csharp' and not 'c_sharp'."
        )
        raise ConfigError(msg)

    undeclared = sorted({language for language in languages if language not in MEDIA_TYPES})
    if undeclared:
        msg = (
            f"grammar language key(s) {undeclared} name real grammars, but manicule declares "
            f"no media type for them, so no document could route to them. Supported keys: "
            f"{list(DECLARED_LANGUAGES)}."
        )
        raise ConfigError(msg)

    return tuple(sorted(set(languages)))


def _supported_names() -> frozenset[str]:
    """Every grammar name the installed pack knows, read offline from the wheel."""
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    global _supported  # noqa: PLW0603 - a one-shot cache of an immutable value
    if _supported is None:
        _supported = frozenset(get_args(pack.SupportedLanguage))
    return _supported


def language_for_media_type(media_type: str) -> str | None:
    """The declared language a media type routes to, or ``None`` when none does."""
    return _LANGUAGES_BY_MEDIA_TYPE.get(media_type)


def scope_separator(language: str) -> str:
    """How ``language`` joins the parts of a qualified name."""
    return SCOPE_SEPARATORS.get(language, DEFAULT_SCOPE_SEPARATOR)


def cache_directory() -> Path:
    """Where the pack keeps grammars for the running configuration."""
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    return Path(pack.cache_dir())


def _remember_default_cache_directory() -> str:
    """The pack's own per-user cache directory, read once while it is still in force.

    Read rather than recomputed. The path carries the pack release and the platform's cache
    convention, so deriving it here would duplicate upstream's rules and go wrong on the
    release that changes them. Reading it on the first call is safe because every override
    manicule applies goes through :func:`configure_pack`, so the pack is still on its default
    the first time this runs.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    global _default_cache_dir  # noqa: PLW0603 - a one-shot cache of an immutable value
    if _default_cache_dir is None:
        _default_cache_dir = pack.cache_dir()
    return _default_cache_dir


def configure_pack(
    languages: Sequence[str],
    *,
    cache_dir: Path | None = None,
    manifest_url: str | None = None,
) -> None:
    """Point the pack at a cache and a manifest, without fetching anything.

    Both overrides exist for deployments rather than for taste: a container image wants the
    cache inside the image, and an air-gapped site wants the manifest on its own mirror.

    Passing ``None`` for either restores the pack's default, and doing so takes explicit
    work: the pack's own configuration call is **additive**, so handing it a null cache
    directory leaves the previous override in force rather than clearing it. "The default"
    therefore means the directory the pack reported before this module changed anything,
    captured once. It matters because the pack keeps one registry per process, so a call
    that meant to reset would otherwise silently inherit whatever ran before it.

    Args:
        languages: The declared set, recorded with the pack so its own prefetch entry points
            act on the same list this module validates.
        cache_dir: Where grammars live. ``None`` uses the per-user cache.
        manifest_url: Where the grammar manifest is fetched from. ``None`` uses the public
            one.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    default_cache = _remember_default_cache_directory()
    if manifest_url is None:
        os.environ.pop(MANIFEST_URL_ENV, None)
    else:
        os.environ[MANIFEST_URL_ENV] = manifest_url
    pack.configure(
        pack.PackConfig(
            # The default is passed back explicitly rather than as ``None``. The pack reads
            # an absent ``cache_dir`` as "keep whatever is already in force", so a call
            # meaning "go back to the default" would silently keep the previous override —
            # and every later call would then report the languages of a directory nobody
            # asked for. Verified against the installed pack, not assumed.
            cache_dir=str(cache_dir) if cache_dir is not None else default_cache,
            languages=list(languages),
        )
    )
    # The compiled query and parser caches are keyed by language, not by cache directory, so
    # a grammar loaded under the previous configuration would otherwise outlive it and this
    # module would report a language as available that the new cache does not contain.
    _PARSERS.clear()
    _QUERIES.clear()


def is_available(language: str) -> bool:
    """Whether ``language``'s grammar is in the cache, checked without any network access."""
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    return language in set(pack.downloaded_languages())


def missing_grammars(languages: Sequence[str]) -> tuple[str, ...]:
    """Which of ``languages`` have no grammar in the cache. Sorted, and no network access.

    What ``manicule doctor`` reports — as its ``grammars`` check, which reads this against the
    *configured* cache directory rather than the per-user default, because those differ on
    exactly the deployments that care — and what :func:`prefetch` acts on.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    present = set(pack.downloaded_languages())
    return tuple(sorted(language for language in languages if language not in present))


def bundle_status(bundle_dir: Path | None = None) -> str:
    """One line describing what this install has to seed from offline.

    What ``manicule doctor`` prints — in its ``grammars`` check, whenever a bundle is installed
    or a grammar is absent — and what every pre-seed failure quotes. An operator whose
    air-gapped host will not parse code needs to know whether a bundle was found, which pack
    release it was built for, and which languages it carries — because "no grammars and no
    network" has three different fixes depending on that answer.

    Raises:
        GrammarBundleError: A bundle exists and cannot be used. Reporting is not a reason to
            downgrade that to a shrug: a bundle that is present, wrong, and described as
            absent is how a host ends up waiting for a mirror it does not have.
    """
    from manicule.parsers import grammar_bundle  # noqa: PLC0415 - lazy, see module docstring

    bundle = grammar_bundle.resolve(bundle_dir)
    if bundle is None:
        return grammar_bundle.describe_search_path(bundle_dir)
    return (
        f"offline grammar bundle at {bundle.root}: tree-sitter-language-pack "
        f"{bundle.pack_version}, {bundle.platform}, {len(bundle.languages)} languages "
        f"({', '.join(bundle.languages)})"
    )


def prefetch(languages: Sequence[str], *, bundle_dir: Path | None = None) -> tuple[str, ...]:
    """Seed every declared grammar that is not already cached. Returns what was seeded.

    The entry point behind ``manicule init`` and ``manicule doctor --fix``, and the reason
    nothing downloads during ingest: pre-seeding is a step an operator runs and can see fail,
    where a lazy fetch is a step that succeeds on one machine and not on another.

    Both commands reach it through one method — ``ApplicationService._grammar_check(fix=True)``
    — so ``init`` and the repair cannot come to mean different things, and both report what
    they seeded rather than doing it silently. ``init`` treats a failure here as a note rather
    than an error: the configuration file is written by then, and a machine that will only ever
    index Markdown is entitled to finish installing without a grammar in sight.

    **The offline bundle is consulted first, and the network only for what it did not supply.**
    That order is what makes an air-gapped install work rather than merely fail politely: a
    host carrying a bundle for the declared set never reaches the fetch at all, so it needs no
    route to the public manifest and no internal mirror. It is also the cheaper order on a
    machine that *does* have network, since a file copy beats a release download.

    Args:
        languages: The declared set. Validated first, so this cannot half-seed a typo.
        bundle_dir: An offline bundle to seed from. ``None`` looks in the places
            :func:`manicule.parsers.grammar_bundle.locate` describes; a bundle that is present
            and unusable is an error rather than a silent fall through to the network.

    Returns:
        The languages actually seeded — from the bundle, from the network, or both — in sorted
        order. Empty when everything was cached, which is the steady state and is what makes
        calling this on every start cheap.

    Raises:
        ConfigError: A declared key is not in the manifest.
        GrammarBundleError: A bundle was found and is unusable.
        GrammarFetchError: The fetch failed — no route to the manifest, a mirror that does
            not have it, a checksum mismatch — **or it reported success and the grammar is
            still not in the cache**, which is checked rather than assumed. See below.
    """
    from manicule.parsers import grammar_bundle  # noqa: PLC0415 - lazy, see module docstring

    wanted = validate_languages(languages)
    absent = missing_grammars(wanted)
    if not absent:
        return ()

    bundle = grammar_bundle.resolve(bundle_dir)
    seeded: tuple[str, ...] = ()
    if bundle is not None:
        seeded = bundle.seed(absent, cache_directory())
        absent = missing_grammars(absent)
    if not absent:
        return seeded

    _fetch(absent, bundle_dir)

    # The pack's own prefetch returns without error for a language it has already loaded into
    # its process-global registry, **even when the configured cache does not contain it** —
    # observed, not assumed. So a container build that loads one Python file and then
    # pre-seeds into an image directory is told it succeeded and ships an image with no
    # grammars in it, and the failure surfaces on an air-gapped host as a refusal to parse
    # code that worked on the machine that built it. Asking the cache afterwards costs a
    # directory listing and turns that into an error where it can still be acted on.
    still_missing = missing_grammars(absent)
    if still_missing:
        detail = (
            f"the grammar pack reported success but {list(still_missing)} is still not in the cache"
        )
        raise GrammarFetchError(_fetch_failure(still_missing, detail, bundle_dir))
    return tuple(sorted({*seeded, *absent}))


def _fetch(languages: Sequence[str], bundle_dir: Path | None) -> None:
    """Download ``languages`` into the configured cache, retrying a dropped transfer.

    The retry is deliberately the dumbest one that fixes the observed failure: the same request
    again, after a bounded wait, up to :data:`FETCH_RETRY_DELAYS` times. It does not recompute
    what is still outstanding between attempts — the pack skips what it already has, so the
    saving would be small, and asking the cache mid-retry would put a second reading of "what
    is missing" next to the authoritative one :func:`prefetch` performs afterwards. One place
    decides whether a seed worked.

    **Success here is not success.** Every path out of this function that does not raise returns
    to that check, which is what makes a retry safe to add: there is no attempt count at which
    this reports a set of grammars it did not actually land.

    Raises:
        GrammarFetchError: Every attempt failed. The message is built from the *last* failure,
            because that is the one describing the state the host is in now, and it names the
            attempt count so that a genuine outage is not read as a single unlucky request.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    delays = FETCH_RETRY_DELAYS
    attempts = len(delays) + 1
    for attempt in range(1, attempts + 1):
        try:
            pack.prefetch(list(languages))
        except (pack.Error, OSError, RuntimeError) as exc:
            # RuntimeError is in this tuple because the native layer maps some download failures
            # to the pack's own exception hierarchy and others to a bare RuntimeError; catching
            # only the former would let an unreachable mirror surface as an unhandled crash.
            if attempt == attempts:
                plural = "" if attempts == 1 else "s"
                detail = f"{exc} (after {attempts} attempt{plural})"
                raise GrammarFetchError(_fetch_failure(languages, detail, bundle_dir)) from exc
            time.sleep(delays[attempt - 1])
        else:
            return


def _fetch_failure(languages: Sequence[str], detail: str, bundle_dir: Path | None = None) -> str:
    """One message for every way a pre-seed can fail, naming the mirror and the bundle.

    The bundle is quoted because this error is read almost exclusively on hosts that have no
    network: "connection refused" sends an operator to their firewall, when the actionable
    fact is that the languages they need were not in the bundle they copied over — or that
    there is no bundle at all.
    """
    from manicule.parsers import grammar_bundle  # noqa: PLC0415 - lazy, see module docstring

    url = os.environ.get(MANIFEST_URL_ENV, "the pack's default manifest URL")
    try:
        offline = bundle_status(bundle_dir)
    except grammar_bundle.GrammarBundleError as exc:  # pragma: no cover - resolve() raised first
        offline = f"the offline grammar bundle is unusable: {exc}"
    return (
        f"could not seed grammars for {list(languages)} from {url} into "
        f"{cache_directory()}: {detail} — and {offline}. Set {MANIFEST_URL_ENV} to an internal "
        f"mirror if this host has a route to one, or narrow the declared language set. "
        f"docs/parsing.md §8.1 covers building a bundle for a host with neither."
    )


def grammar_versions(languages: Sequence[str]) -> dict[str, str]:
    """Grammar version per declared language, for ``ChunkFingerprint.grammars``.

    A grammar upgrade changes parse trees, changed trees change split points, and changed
    split points mean stored embeddings that no longer correspond to the chunks that would
    be produced today. Recording this per language is what lets a grammar bump invalidate
    the documents in that language instead of the corpus.

    The value is the **pack release**, for every declared language, and that is the honest
    answer rather than a placeholder. Two alternatives were rejected:

    - *A per-grammar version number.* The pack does not carry one. ``Language.abi_version``
      is the tree-sitter ABI (14 for every grammar in this release, and it moves for reasons
      unrelated to a grammar's content) and ``Language.semantic_version`` is ``None`` for
      every grammar here — verified, not assumed.
    - *A hash of the grammar's shared library.* It would be genuinely per-language, and it
      would also differ between macOS and Linux for identical grammar source, so moving a
      corpus between machines would invalidate it for no reason. ``fingerprints.py`` is
      explicit that fields describing the producer without affecting its output stay out of
      identity; a platform-dependent hash is worse than that, since it *looks* like output.

    Grammars ship as one platform bundle per pack release, so every grammar moves together
    and the release version describes all of them exactly. Keeping the map per language
    means a pack that later versions grammars individually needs no schema change.

    Args:
        languages: The declared set.

    Returns:
        ``{language: version}`` over the declared set, independent of what is cached. It has
        to be independent: a map that shrank when a grammar was missing would make the
        fingerprint depend on cache state, which is the hazard this module exists to close.
    """
    version = pack_version()
    return dict.fromkeys(validate_languages(languages), version)


_PARSERS: dict[str, Parser] = {}
_QUERIES: dict[str, Query | None] = {}


def load_parser(language: str) -> Parser:
    """A parser for ``language``, or a refusal if its grammar is not cached.

    Args:
        language: A declared, validated language key.

    Returns:
        The pack's parser for that grammar, cached per language for the life of the process.

    Raises:
        GrammarUnavailableError: The grammar is not in the cache. Checked before the pack is
            asked for a parser, because asking would fetch it — and a fetch here is the
            corpus-consistency hazard arriving through the back door.
        GrammarUnusableError: The grammar is in the cache and will not load — the case an
            offline bundle makes reachable, since a bundle is the one artifact that can put a
            library built for another platform on this machine.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    cached = _PARSERS.get(language)
    if cached is not None:
        return cached
    if not is_available(language):
        raise GrammarUnavailableError(language)
    try:
        parser = pack.get_parser(language)
    except (pack.Error, OSError) as exc:
        raise GrammarUnusableError(language, str(exc)) from exc
    _PARSERS[language] = parser
    return parser


def tags_query_source(language: str) -> str | None:
    """The pack's tags query for ``language`` as source text, or ``None`` if it has none.

    Resolves offline and without a grammar, which is what makes it usable for symbols at
    all: a query set that varied by machine would be the same corpus-consistency hazard the
    declared language set closes.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring

    return pack.get_tags_query(language)


def tags_query(language: str) -> Query | None:
    """The pack's tags query for ``language``, compiled, or ``None`` if it has none.

    Compiled once per process because compiling is the expensive part, while running it over
    a tree is not.

    Raises:
        GrammarUnavailableError: The grammar is not cached. A query compiles against a
            language, so this cannot answer before :func:`load_parser` would have refused.
        GrammarUnusableError: The grammar is cached and will not load, for the same reasons
            :func:`load_parser` gives. Wrapped here too rather than only there, because a
            symbol lookup reaching the pack directly would otherwise turn a broken library into
            an exception with no language in it.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - lazy, see module docstring
    from tree_sitter import Query  # noqa: PLC0415 - lazy, see module docstring

    if language in _QUERIES:
        return _QUERIES[language]
    source = tags_query_source(language)
    if source is None:
        _QUERIES[language] = None
        return None
    if not is_available(language):
        raise GrammarUnavailableError(language)
    try:
        grammar = pack.get_language(language)
    except (pack.Error, OSError) as exc:
        raise GrammarUnusableError(language, str(exc)) from exc
    query = Query(grammar, source)
    _QUERIES[language] = query
    return query
