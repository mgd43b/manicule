"""The offline grammar bundle: grammars that arrive with the install rather than over HTTP.

:mod:`manicule.parsers.grammars` closes the ordinary case — a declared language set, a
pre-seed step, and a refusal instead of a fallback. It does not close the air-gapped one.
``prefetch`` still fetches, and a host with no route to the grammar release has nothing to
fetch *from*: ``manicule init`` reports a pre-seed it could not complete, ``manicule doctor``
reports the grammars as absent, and the code parser refuses every source document afterwards.
Pointing the manifest at an internal mirror assumes somebody has a mirror.

A bundle is the answer that assumes nothing: **a directory of grammar libraries plus a
manifest describing exactly which release they came from**, produced once by
``tools/build_grammar_bundle.py`` and carried to the air-gapped host by whatever moves the
install itself. Seeding from it is a file copy, so it works with no network, with an empty
cache, and with a cache directory that resolves somewhere nobody expected.

**The bundle is a source, never the cache.** :meth:`GrammarBundle.seed` copies libraries out
of it into the configured cache and the ordinary load path takes over from there. Two reasons,
and both were reachable by the other design:

- A bundle installed under ``site-packages`` is **read-only** on any sensibly built image.
  Making it the cache would mean the pack writing into it the first time a language outside
  the bundle is asked for.
- Seeding leaves exactly one directory that answers "which grammars does this machine have",
  which is what ``missing_grammars`` reports and what the pre-seed asserts afterwards. Two
  answers to that question is how a machine ends up believing it has a grammar it cannot load.

The bundle is still usable *directly* — its library directory is a valid cache directory, and
a read-only container that never wants a copy can be pointed at it — but nothing here does
that on its own, because it would make the cache setting depend on where a bundle happened to
be found.

**Everything the bundle records is recorded, not inferred.** The pack release, the platform,
the file name each language's library actually has, its size and its SHA-256. A bundle that
disagrees with the installed pack, or that was built for another platform, is an error naming
both sides rather than a set of libraries that load into the wrong ABI or do not load at all.
That matters more here than anywhere else in this package: a grammar bundle is the one artifact
in manicule that is *both* platform-specific and version-specific, and the failure mode of
getting it wrong is a parse tree that differs from the one every other machine produces.

**Library file names are discovered, not constructed.** The obvious rule —
``libtree_sitter_{language}`` — is wrong for at least one declared language: ``csharp``'s
library is ``libtree_sitter_c_sharp``. A build-time guess would ship a bundle silently missing
that language, so :func:`build` asks the pack itself which file answers for which language and
writes the answer into the manifest. The seeding path then needs no rule at all.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Final

from manicule.core.errors import ManiculeError

__all__ = [
    "BUNDLE_DIR_ENV",
    "BUNDLE_MODULE",
    "COPYLEFT_PREFIXES",
    "LIBRARY_DIR_NAME",
    "MANIFEST_NAME",
    "PERMISSIVE_LICENCES",
    "SCHEMA_VERSION",
    "BundledGrammar",
    "GrammarBundle",
    "GrammarBundleError",
    "build",
    "check_licence",
    "describe_search_path",
    "library_suffix",
    "licence_of_installed_pack",
    "locate",
    "platform_tag",
    "read",
    "resolve",
]

MANIFEST_NAME: Final = "grammars.json"
"""The bundle's manifest, at the root of the bundle directory."""

LIBRARY_DIR_NAME: Final = "libs"
"""Where the grammar libraries sit inside a bundle.

The same name the pack gives its own cache subdirectory, so the directory can be handed
straight to ``configure(PackConfig(cache_dir=...))`` by a read-only deployment that wants no
copy at all.
"""

SCHEMA_VERSION: Final = 1
"""The manifest layout. Checked on read, so a bundle from a later manicule is refused with
both numbers rather than parsed into the wrong shape."""

BUNDLE_DIR_ENV: Final = "MANICULE_GRAMMAR_BUNDLE"
"""Environment variable naming a bundle directory.

An environment variable rather than a settings field because the pre-seed runs *before* the
thing that would read settings: the grammar pre-seed is a step an image build or an installer
performs against ``manicule.parsers.grammars`` directly, with no configured parser and no
container in existence yet. A caller that does have configuration passes the path explicitly
instead — ``prefetch(..., bundle_dir=...)`` — and that argument wins.

Inside manicule's ``MANICULE_`` namespace deliberately, which the test environment fixture
clears before every test. That is the right treatment for this one: it is deployment
configuration, and a developer's own bundle leaking into the suite would make tests pass
because of what is on their machine. The switch that arms the offline suite in CI is a
different kind of thing and is named outside the namespace for the opposite reason.
"""

BUNDLE_MODULE: Final = "manicule_grammars"
"""An importable module carrying a bundle, for installs that ship one as a distribution.

The bundle then travels through the same channel as every other dependency, so an air-gapped
host that can install manicule at all can install its grammars, and nobody has to remember to
copy a directory to the right place. ``tools/build_grammar_bundle.py --package`` writes such a
distribution; it is built per platform and per pack release, which is why manicule builds one
rather than publishing one. The module needs one thing — a ``bundle`` directory beside its
``__init__`` — and nothing here imports it for any other purpose.
"""

PERMISSIVE_LICENCES: Final[frozenset[str]] = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "ISC",
        "MIT",
        "MIT-0",
        "Unlicense",
        "Zlib",
    }
)
"""Licences a bundled grammar may carry. ``docs/parsing.md`` §12's list, as data."""

COPYLEFT_PREFIXES: Final[tuple[str, ...]] = ("GPL", "AGPL", "LGPL", "MPL", "SSPL", "EUPL", "CDDL")
"""Refused outright, and named separately from "not on the permissive list" so the message
says *why*. manicule is GPL-3.0-or-later, so a copyleft grammar would not be a licence problem
for manicule — it would be one for anyone redistributing the bundle, and this bundle exists to
be redistributed."""

_LICENCE_OPERATORS: Final[frozenset[str]] = frozenset({"AND", "OR", "WITH"})

_PLATFORMS: Final[Mapping[str, str]] = {"darwin": "macos", "linux": "linux", "win32": "windows"}
_MACHINES: Final[Mapping[str, str]] = {
    "aarch64": "aarch64",
    "amd64": "x86_64",
    "arm64": "arm64",
    "x86_64": "x86_64",
}
_LIBRARY_SUFFIXES: Final[Mapping[str, str]] = {"darwin": ".dylib", "win32": ".dll"}


class GrammarBundleError(ManiculeError):
    """A bundle was found and cannot be used.

    Deliberately loud, and deliberately *not* raised when no bundle exists at all. "No bundle
    is installed" is an ordinary state — the machine fetches instead — while "a bundle is
    installed and does not describe this pack, this platform, or these files" is a
    misconfiguration that would otherwise surface as a grammar that loads into the wrong ABI or
    a document that mysteriously will not parse.
    """


def platform_tag() -> str:
    """The platform a bundle is built for and may be used on.

    Grammar libraries are compiled objects, so a bundle is valid for exactly one operating
    system and architecture. The vocabulary matches the grammar release's own — ``macos-arm64``,
    ``linux-x86_64`` — because a human comparing a bundle against the release it came from
    should not have to translate. An unrecognised platform still produces a tag rather than an
    error: both the builder and the reader call this function, so what matters is that the two
    agree, and an exotic platform's bundle is refused on every *other* platform either way.
    """
    system = _PLATFORMS.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    return f"{system}-{_MACHINES.get(machine, machine)}"


def library_suffix() -> str:
    """The file extension a shared library has here: ``.dylib``, ``.dll`` or ``.so``."""
    return _LIBRARY_SUFFIXES.get(sys.platform, ".so")


@dataclass(frozen=True, slots=True)
class BundledGrammar:
    """One language's library, as the manifest records it."""

    language: str
    filename: str
    """The library's own file name, discovered from the pack rather than derived from
    ``language``. ``csharp`` ships as ``libtree_sitter_c_sharp``, and a bundle built on the
    assumption that it does not is a bundle missing C#."""
    sha256: str
    size: int

    def as_json(self) -> dict[str, Any]:
        """The manifest form. Keys sorted by :func:`json.dumps`, so a rebuild diffs cleanly."""
        return {"filename": self.filename, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class GrammarBundle:
    """A validated bundle on disk."""

    root: Path
    pack_version: str
    platform: str
    licence: str
    grammars: Mapping[str, BundledGrammar]

    @property
    def library_dir(self) -> Path:
        """Where the libraries are. Also a valid pack cache directory, read-only included."""
        return self.root / LIBRARY_DIR_NAME

    @property
    def languages(self) -> tuple[str, ...]:
        """What this bundle can seed, sorted."""
        return tuple(sorted(self.grammars))

    def path_for(self, language: str) -> Path:
        """The library file for ``language``.

        Raises:
            GrammarBundleError: The bundle does not carry it. A caller asking for a language
                outside the bundle is asking the wrong bundle, which is worth a message rather
                than a ``KeyError``.
        """
        entry = self.grammars.get(language)
        if entry is None:
            msg = (
                f"the grammar bundle at {self.root} carries {list(self.languages)} and not "
                f"{language!r}. Rebuild it with that language declared."
            )
            raise GrammarBundleError(msg)
        return self.library_dir / entry.filename

    def verify(self) -> None:
        """Hash every library and compare it against the manifest.

        Not done on read. Reading a bundle happens on every pre-seed and hashing tens of
        megabytes to answer "is a bundle present" would make the cheap path expensive, while
        the copy in :meth:`seed` has to read every byte anyway and hashes them on the way past.
        This exists for the deliberate check — a ``doctor`` command, or a build step confirming
        what it just wrote — where reading the whole bundle is the point.

        Raises:
            GrammarBundleError: A library is missing, truncated, or does not hash to what the
                manifest says.
        """
        for language in self.languages:
            self._verified_bytes(language)

    def seed(self, languages: Sequence[str], cache_dir: Path) -> tuple[str, ...]:
        """Copy the libraries for ``languages`` into ``cache_dir``. Returns what was copied.

        The whole air-gapped path, and it is a file copy: no manifest, no HTTP client, no
        release lookup. Languages the bundle does not carry are skipped rather than refused,
        because the caller — :func:`manicule.parsers.grammars.prefetch` — still has the network
        to try for them and is the one that decides whether their absence is fatal.

        Each library is hashed as it is copied and written under a temporary name that is only
        renamed into place once the hash matches. A half-written library under its real name
        would be reported by the pack as a downloaded language and then fail to load, which is
        the "present but unusable" state this module exists to make impossible.

        Args:
            languages: Wanted languages, in any order.
            cache_dir: The pack's configured cache. Created if it does not exist, because an
                air-gapped install's cache directory has never been written to.

        Returns:
            The languages copied, sorted.

        Raises:
            GrammarBundleError: A library is missing from the bundle directory, or its content
                does not match the manifest.
        """
        wanted = sorted(set(languages) & set(self.grammars))
        if not wanted:
            return ()
        cache_dir.mkdir(parents=True, exist_ok=True)
        for language in wanted:
            entry = self.grammars[language]
            destination = cache_dir / entry.filename
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, prefix=f".{entry.filename}.", delete=False
            ) as handle:
                partial = Path(handle.name)
            try:
                partial.write_bytes(self._verified_bytes(language))
                shutil.copymode(self.path_for(language), partial)
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        return tuple(wanted)

    def _verified_bytes(self, language: str) -> bytes:
        """A library's content, checked against the manifest before it is handed back."""
        entry = self.grammars[language]
        path = self.path_for(language)
        try:
            content = path.read_bytes()
        except OSError as exc:
            msg = (
                f"the grammar bundle at {self.root} lists {language!r} as "
                f"{entry.filename} and that file cannot be read: {exc}. Rebuild the bundle "
                f"with tools/build_grammar_bundle.py."
            )
            raise GrammarBundleError(msg) from exc
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != entry.size or digest != entry.sha256:
            msg = (
                f"{path} does not match the grammar bundle manifest: recorded "
                f"{entry.size} bytes / sha256 {entry.sha256}, found {len(content)} bytes / "
                f"sha256 {digest}. The bundle is corrupt or was edited after it was built; "
                f"rebuild it with tools/build_grammar_bundle.py."
            )
            raise GrammarBundleError(msg)
        return content


def locate(explicit: Path | None = None) -> Path | None:
    """Where the bundle is, or ``None`` when this install has none.

    Three sources, highest priority first, and the order is the point: a caller that named a
    path meant that path, a deployment that set the variable meant it for every process it
    starts, and an installed distribution is the default nobody has to configure.

    Args:
        explicit: A path a caller named. Returned unchanged — including when it does not
            exist, so that a typo is an error from :func:`read` rather than a silent fall
            through to some other bundle.

    Returns:
        A bundle directory, or ``None``.
    """
    if explicit is not None:
        return explicit
    configured = os.environ.get(BUNDLE_DIR_ENV)
    if configured:
        return Path(configured)
    return _installed_bundle()


def _installed_bundle() -> Path | None:
    """The bundle carried by an installed :data:`BUNDLE_MODULE` distribution, if there is one.

    Resolved through the module's spec rather than by importing it, because a data-only
    distribution has nothing worth executing and an import failure inside it would be reported
    as "no bundle installed" — which is the quiet degradation this whole module is against.
    """
    try:
        spec = find_spec(BUNDLE_MODULE)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).parent / "bundle"
    return root if root.is_dir() else None


def describe_search_path(explicit: Path | None = None) -> str:
    """Where a bundle was looked for, for an error message that has to be actionable.

    An operator seeing "no grammars and no network" needs to know which three places were
    checked, because the fix is to put a bundle in one of them.
    """
    if explicit is not None:
        return f"no usable bundle at {explicit}"
    configured = os.environ.get(BUNDLE_DIR_ENV)
    if configured:
        return f"no usable bundle at {configured} (from {BUNDLE_DIR_ENV})"
    return (
        f"no offline grammar bundle is installed: {BUNDLE_DIR_ENV} is unset and no "
        f"{BUNDLE_MODULE} distribution carries one. Build one on a machine with network "
        f"access using tools/build_grammar_bundle.py and copy it to this host"
    )


def read(root: Path) -> GrammarBundle:
    """Read and validate the bundle at ``root``.

    Every check here answers a way a bundle can be wrong while looking right. A manifest from a
    different schema parses into the wrong shape. A manifest from a different pack release
    describes grammars whose parse trees are not the ones this install's fingerprint claims. A
    bundle built on another platform contains libraries that either fail to load or, worse,
    load and are refused later by something less specific. A truncated library is reported by
    the pack as a language it has, and then fails at ``get_parser`` with "not found".

    Sizes are checked here and hashes are not: a size check is a ``stat`` and catches the
    truncation case, while hashing is deferred to the copy in :meth:`GrammarBundle.seed`, which
    reads every byte regardless.

    Raises:
        GrammarBundleError: Anything above.
    """
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = (
            f"{root} is not a grammar bundle: it has no {MANIFEST_NAME}. Build one with "
            f"tools/build_grammar_bundle.py, or unset {BUNDLE_DIR_ENV} to fetch instead."
        )
        raise GrammarBundleError(msg) from exc
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"the grammar bundle manifest {manifest_path} could not be read: {exc}"
        raise GrammarBundleError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"the grammar bundle manifest {manifest_path} is not an object"
        raise GrammarBundleError(msg)
    manifest: Mapping[str, Any] = raw

    schema = manifest.get("schema_version")
    if schema != SCHEMA_VERSION:
        msg = (
            f"the grammar bundle at {root} declares manifest schema {schema!r}, and this "
            f"manicule reads schema {SCHEMA_VERSION}. Rebuild the bundle with this version's "
            f"tools/build_grammar_bundle.py."
        )
        raise GrammarBundleError(msg)

    installed = _installed_pack_version()
    if manifest.get("pack_version") != installed:
        msg = (
            f"the grammar bundle at {root} was built for tree-sitter-language-pack "
            f"{manifest.get('pack_version')!r} and this install has {installed!r}. Grammars "
            f"ship as one bundle per release, so the two describe different parse trees; "
            f"rebuild the bundle against the installed pack rather than mixing them."
        )
        raise GrammarBundleError(msg)

    here = platform_tag()
    if manifest.get("platform") != here:
        msg = (
            f"the grammar bundle at {root} was built for {manifest.get('platform')!r} and "
            f"this machine is {here!r}. Grammar libraries are compiled objects; build the "
            f"bundle on this platform."
        )
        raise GrammarBundleError(msg)

    bundle = GrammarBundle(
        root=root,
        pack_version=str(manifest["pack_version"]),
        platform=here,
        # Checked, not merely recorded. A bundle is redistributed by hand, so the manifest it
        # arrives with is the only licence statement the receiving machine has, and a manifest
        # is a text file somebody can edit. Re-asserting it here costs a string split and means
        # the terms cannot be widened between the build and the install.
        licence=check_licence(str(manifest.get("licence", ""))),
        grammars=_entries(manifest, manifest_path),
    )
    _check_sizes(bundle)
    return bundle


def _entries(manifest: Mapping[str, Any], path: Path) -> Mapping[str, BundledGrammar]:
    """The manifest's language table, or an error naming the file rather than a ``KeyError``.

    Every conversion happens here, inside one ``try``. A size recorded as ``"large"`` is a
    corrupt manifest and must read as one; leaving the ``int()`` outside would let it escape as
    a bare ``ValueError`` from somewhere with no file name in it.
    """
    languages = manifest.get("languages")
    if not isinstance(languages, dict) or not languages:
        msg = f"the grammar bundle manifest {path} lists no languages"
        raise GrammarBundleError(msg)
    entries: dict[str, Any] = languages
    try:
        return {
            str(language): BundledGrammar(
                language=str(language),
                filename=str(entry["filename"]),
                sha256=str(entry["sha256"]),
                size=int(entry["size"]),
            )
            for language, entry in entries.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"the grammar bundle manifest {path} has a malformed language entry: {exc}"
        raise GrammarBundleError(msg) from exc


def _check_sizes(bundle: GrammarBundle) -> None:
    """Every library present, and every one the length the manifest recorded."""
    wrong: list[str] = []
    for language in bundle.languages:
        entry = bundle.grammars[language]
        path = bundle.library_dir / entry.filename
        try:
            size = path.stat().st_size
        except OSError:
            wrong.append(f"{language} ({entry.filename}: missing)")
            continue
        if size != entry.size:
            wrong.append(f"{language} ({entry.filename}: {size} bytes, expected {entry.size})")
    if wrong:
        msg = (
            f"the grammar bundle at {bundle.root} is incomplete: {'; '.join(wrong)}. It was "
            f"built wrong or damaged in transit; rebuild it with tools/build_grammar_bundle.py."
        )
        raise GrammarBundleError(msg)


def resolve(explicit: Path | None = None) -> GrammarBundle | None:
    """The usable bundle for this install, or ``None`` when there is no bundle at all.

    The one call the pre-seed makes. ``None`` means "nothing to seed from, use the network";
    an exception means "a bundle is here and is wrong", which is never quietly downgraded to
    the first, because a bundle that is present and ignored is how an air-gapped host ends up
    fetching from a mirror it does not have.
    """
    root = locate(explicit)
    if root is None:
        return None
    return read(root)


def _installed_pack_version() -> str:
    """The installed grammar pack release, read from distribution metadata."""
    from manicule.parsers.grammars import pack_version  # noqa: PLC0415 - avoids an import cycle

    return pack_version()


def check_licence(expression: str) -> str:
    """Assert an SPDX licence expression is one a bundle may redistribute. Returns it.

    The upstream pack states that every grammar it carries is permissively licensed and that
    copyleft grammars are not accepted. This asserts the policy rather than trusting it, at the
    moment a bundle is built — which is the moment redistribution starts, and the last one at
    which a change in that policy is cheap to notice.

    Args:
        expression: An SPDX expression, e.g. ``"MIT"`` or ``"Apache-2.0 OR MIT"``.

    Returns:
        The expression, unchanged, so a caller can record what it asserted rather than
        re-deriving it.

    Raises:
        GrammarBundleError: A term is copyleft, or is not on the permissive list. The two are
            separate messages because they mean different things: the first is a licence
            manicule refuses to put in a redistributable bundle, the second is a licence nobody
            here has assessed, and "add it to the list" is only the right answer to one of them.
    """
    terms = [term for term in re.split(r"[\s()]+", expression.strip()) if term]
    if not terms:
        msg = (
            "no grammar licence is declared, so the bundle would be redistributed under "
            "unknown terms. A bundle built by tools/build_grammar_bundle.py always records "
            "one; a manifest without it has been edited or was written by something else."
        )
        raise GrammarBundleError(msg)
    named = [term for term in terms if term.upper() not in _LICENCE_OPERATORS]
    copyleft = [term for term in named if term.upper().startswith(COPYLEFT_PREFIXES)]
    if copyleft:
        msg = (
            f"the grammar licence {expression!r} includes the copyleft term(s) {copyleft}. "
            f"docs/parsing.md §12 refuses a copyleft grammar bundle: it would put a "
            f"source-distribution obligation on everyone who copies the bundle to an "
            f"air-gapped host, which is an obligation manicule would be imposing rather than "
            f"accepting."
        )
        raise GrammarBundleError(msg)
    unknown = sorted({term for term in named if term not in PERMISSIVE_LICENCES})
    if unknown:
        msg = (
            f"the grammar licence {expression!r} contains {unknown}, which is not on the list "
            f"of licences assessed for redistribution ({sorted(PERMISSIVE_LICENCES)}). Assess "
            f"it against docs/parsing.md §12 before adding it, rather than adding it to make "
            f"this pass."
        )
        raise GrammarBundleError(msg)
    return expression


def licence_of_installed_pack() -> str:
    """The grammar pack's declared SPDX licence expression.

    The strongest licence statement the installed artifact supports, and the reason is worth
    recording where the bundle is built: **the pack enumerates no per-grammar licences.** Its
    manifest carries a group and a size per language, the wheel carries no licence files for
    the grammars, and the SBOM beside it describes the Rust build dependencies of the native
    extension. So the assertion is over the distribution that publishes them under a stated
    permissive-only policy, and the bundle records which expression was asserted so that a
    later release changing it is a build failure rather than a discovery.
    """
    from importlib.metadata import metadata  # noqa: PLC0415 - only needed when building

    from manicule.parsers.grammars import PACK_DISTRIBUTION  # noqa: PLC0415 - import cycle

    fields = metadata(PACK_DISTRIBUTION)
    return fields["License-Expression"] or fields["License"] or ""


def build(
    languages: Sequence[str],
    destination: Path,
    *,
    source: Path,
) -> GrammarBundle:
    """Write a bundle for ``languages`` into ``destination``, from libraries in ``source``.

    ``source`` is a populated pack cache: the machine that builds a bundle is a machine with
    network access, which has already fetched the declared set. Building is therefore a copy, a
    hash and a licence assertion — never a download, so a build cannot half-succeed against a
    flaky mirror and produce a bundle missing a language nobody notices until an air-gapped
    host refuses a document.

    **Which file belongs to which language is asked of the pack, not computed.** Every
    candidate is offered to the pack alone in an empty directory and the pack says which
    language it answers for. That is slower than an f-string and it is the difference between
    a bundle with C# in it and a bundle without: ``csharp``'s library is
    ``libtree_sitter_c_sharp``.

    Args:
        languages: The languages to bundle. Validated against the pack's manifest first, so a
            typo fails before anything is written.
        destination: Created if absent. An existing bundle in it is overwritten.
        source: A pack cache directory holding the compiled libraries.

    Returns:
        The bundle that was written, re-read from disk — so the return value is proof the
        manifest describes what is actually there rather than what was intended.

    Raises:
        ConfigError: A language is not one manicule declares.
        GrammarBundleError: The pack's licence is not redistributable, ``source`` holds no
            library for a requested language, or the bundle fails to read back.
    """
    from manicule.parsers.grammars import validate_languages  # noqa: PLC0415 - import cycle

    wanted = validate_languages(languages)
    licence = check_licence(licence_of_installed_pack())
    found = _discover_libraries(wanted, source)

    library_dir = destination / LIBRARY_DIR_NAME
    library_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, BundledGrammar] = {}
    for language in wanted:
        origin = found[language]
        content = origin.read_bytes()
        shutil.copy2(origin, library_dir / origin.name)
        entries[language] = BundledGrammar(
            language=language,
            filename=origin.name,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pack_version": _installed_pack_version(),
        "platform": platform_tag(),
        "licence": licence,
        "languages": {language: entry.as_json() for language, entry in entries.items()},
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle = read(destination)
    bundle.verify()
    return bundle


def _discover_libraries(languages: Sequence[str], source: Path) -> dict[str, Path]:
    """Ask the pack which file in ``source`` is which language's grammar.

    Two passes, because a probe is a directory of symlinks and a pack reconfiguration, and
    doing 24 of them when 23 are answerable by name is waste rather than rigour. The name is
    still *confirmed* by a probe — an unconfirmed name match is the guess this function exists
    to avoid.

    Raises:
        GrammarBundleError: A requested language has no library in ``source``. Never a partial
            bundle: a bundle silently missing one language is a host that refuses one language,
            which looks like a routing bug rather than a build mistake.
    """
    candidates = sorted(path for path in source.glob(f"*{library_suffix()}") if path.is_file())
    if not candidates:
        msg = (
            f"no grammar libraries (*{library_suffix()}) in {source}. Pre-seed the declared "
            f"set on this machine first — a bundle is built from a populated cache, never "
            f"downloaded."
        )
        raise GrammarBundleError(msg)

    found: dict[str, Path] = {}
    unclaimed = set(candidates)
    with _pack_configuration_restored():
        for language in languages:
            expected = source / f"libtree_sitter_{language}{library_suffix()}"
            if expected in unclaimed and _answers_for(expected, language):
                found[language] = expected
                unclaimed.discard(expected)
        for language in languages:
            if language in found:
                continue
            for candidate in sorted(unclaimed):
                if _answers_for(candidate, language):
                    found[language] = candidate
                    unclaimed.discard(candidate)
                    break

    missing = [language for language in languages if language not in found]
    if missing:
        msg = (
            f"no grammar library in {source} answers for {missing}. Pre-seed those languages "
            f"before building the bundle; a bundle that silently omits a language becomes a "
            f"host that refuses it."
        )
        raise GrammarBundleError(msg)
    return found


def _answers_for(candidate: Path, language: str) -> bool:
    """Whether the pack reports ``language`` when ``candidate`` is the only library present.

    The authoritative answer, because it is the pack's own resolution rather than a rule about
    file names. Isolation is what makes it authoritative: a directory holding one library
    cannot report a language that some other file supplied.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - a parsing extra, not core

    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / candidate.name
        try:
            probe.symlink_to(candidate)
        except OSError:  # pragma: no cover - filesystems without symlinks
            shutil.copy2(candidate, probe)
        pack.configure(pack.PackConfig(cache_dir=directory, languages=[language]))
        return language in set(pack.downloaded_languages())


@contextlib.contextmanager
def _pack_configuration_restored() -> Generator[None]:
    """Put the pack back on the configuration it had, afterwards.

    The pack keeps one registry per process and probing repoints it two dozen times. Without
    this, building a bundle inside a running process would leave the pack looking at a
    temporary directory that no longer exists — and every later ``is_available`` call would
    answer ``False`` for grammars sitting right there.

    What is restored is **what was in force**, read back from the pack and the environment,
    rather than manicule's defaults. Restoring the default would be the same class of bug as
    the one being guarded against: a container that configured an image-local cache and then
    built a bundle would silently be moved back to the per-user cache, and would look for its
    grammars somewhere they have never been.
    """
    import tree_sitter_language_pack as pack  # noqa: PLC0415 - a parsing extra, not core

    from manicule.parsers import grammars  # noqa: PLC0415 - avoids an import cycle

    cache = Path(pack.cache_dir())
    manifest_url = os.environ.get(grammars.MANIFEST_URL_ENV)
    try:
        yield
    finally:
        grammars.configure_pack(
            grammars.DECLARED_LANGUAGES, cache_dir=cache, manifest_url=manifest_url
        )
