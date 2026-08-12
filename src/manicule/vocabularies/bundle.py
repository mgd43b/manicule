"""The offline vocabulary bundle: BPE vocabularies that arrive with the install.

:mod:`manicule.vocabularies.store` closes the ordinary case — a pre-seed step, and a refusal
instead of a lazy download. It does not close the air-gapped one. ``prefetch`` still fetches,
and a host with no route to ``openaipublic.blob.core.windows.net`` has nothing to fetch
*from*, so the pre-seed fails there and every question afterwards is refused.

A bundle is the answer that assumes nothing: **a directory of vocabulary files plus a manifest
describing exactly which URL and digest each one came from**, produced once by
``tools/build_vocabulary_bundle.py`` on a machine with network access and carried to the
air-gapped host by whatever moves the install itself. Seeding from it is a file copy, so it
works with no network, with an empty cache, and with a cache directory that resolves somewhere
nobody expected.

This is deliberately the same shape as :mod:`manicule.parsers.grammar_bundle`, and where it
differs it differs for a reason worth stating:

- **No platform tag.** A grammar is a compiled shared object and a bundle of them is valid for
  exactly one OS and architecture. A ``.tiktoken`` file is a text table of base64 tokens and
  ranks; the same bytes are correct on every machine, and a platform check would force one
  bundle per architecture to carry identical content.
- **No refusal on a ``tiktoken`` version mismatch.** Grammars ship as one bundle per pack
  release because the release decides the parse trees. Vocabularies do not move with the
  library: ``cl100k_base.tiktoken`` has been the same file across ``tiktoken`` releases, and
  refusing a bundle because the library was bumped would force a rebuild that changes nothing.
  What *is* checked is stronger and is the thing that actually matters — the bundle must carry
  the URL the installed ``tiktoken`` asks for, with the digest that ``tiktoken`` declares for
  it. A release that changed either is caught by that check; one that changed neither needs no
  new bundle. The version is recorded for provenance, and reported, but is not a gate.
- **No licence assertion, and no vocabulary bytes in manicule's own distribution.** The
  grammar pack publishes a permissive-only policy that a bundle can assert against. OpenAI
  publishes these files with no SPDX expression at all, so there is nothing here to assert and
  manicule will not redistribute what it cannot describe. The manifest records the URL each
  file came from instead, so whoever *does* carry a bundle can see precisely what is in it.

**The bundle is a source, never the cache**, for the two reasons the grammar bundle gives: a
bundle installed under ``site-packages`` is read-only on any sensibly built image, and seeding
leaves exactly one directory that answers "which vocabularies does this machine have". It is
still usable directly — the vocabulary directory is laid out as ``tiktoken``'s own cache, file
names and all, so a read-only deployment can point :data:`~manicule.vocabularies.store
.CACHE_DIR_ENV` straight at it — but nothing here does that on its own, because it would make
the cache setting depend on where a bundle happened to be found.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Final, cast

from manicule.core.errors import ManiculeError
from manicule.vocabularies.store import Blob, blobs_for, cache_key, cache_path

__all__ = [
    "BUNDLE_DIR_ENV",
    "BUNDLE_MODULE",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "VOCABULARY_DIR_NAME",
    "BundledVocabulary",
    "VocabularyBundle",
    "VocabularyBundleError",
    "build",
    "describe_search_path",
    "locate",
    "read",
    "resolve",
]

MANIFEST_NAME: Final = "vocabularies.json"
"""The bundle's manifest, at the root of the bundle directory."""

VOCABULARY_DIR_NAME: Final = "vocab"
"""Where the vocabulary files sit inside a bundle.

Laid out exactly as ``tiktoken``'s cache is — each file named with ``tiktoken``'s own cache
key — so the directory can be handed straight to ``TIKTOKEN_CACHE_DIR`` by a read-only
deployment that wants no copy at all.
"""

SCHEMA_VERSION: Final = 1
"""The manifest layout. Checked on read, so a bundle from a later manicule is refused with
both numbers rather than parsed into the wrong shape."""

BUNDLE_DIR_ENV: Final = "MANICULE_VOCABULARY_BUNDLE"
"""Environment variable naming a bundle directory.

An environment variable rather than a settings field because the pre-seed runs *before* the
thing that would read settings: it is a step an image build or an installer performs against
this package directly, with no configured retriever and no container in existence yet. A
caller that does have configuration passes the path explicitly instead —
``prefetch(..., bundle_dir=...)`` — and that argument wins.

Inside manicule's ``MANICULE_`` namespace deliberately, which the test environment fixture
clears before every test: it is deployment configuration, and a developer's own bundle leaking
into the suite would make tests pass because of what is on their machine.
"""

BUNDLE_MODULE: Final = "manicule_vocabularies"
"""An importable module carrying a bundle, for installs that ship one as a distribution.

The bundle then travels through the same channel as every other dependency, so an air-gapped
host that can install manicule at all can install its vocabularies, and nobody has to remember
to copy a directory to the right place. ``tools/build_vocabulary_bundle.py --package`` writes
such a distribution. The module needs one thing — a ``bundle`` directory beside its
``__init__`` — and nothing here imports it for any other purpose.
"""


class VocabularyBundleError(ManiculeError):
    """A bundle was found and cannot be used.

    Deliberately loud, and deliberately *not* raised when no bundle exists at all. "No bundle
    is installed" is an ordinary state — the machine pre-seeds over the network instead —
    while "a bundle is installed and does not describe these files" is a misconfiguration that
    would otherwise surface as a question that cannot be answered.
    """


@dataclass(frozen=True, slots=True)
class BundledVocabulary:
    """One vocabulary file, as the manifest records it."""

    url: str
    """Where ``tiktoken`` would have fetched it. Both the provenance a redistributor needs and
    the identity ``tiktoken`` keys its cache by."""
    sha256: str
    """The digest of the bytes carried here, which :func:`build` asserts is the digest the
    installed ``tiktoken`` declares for this URL."""
    size: int

    @property
    def filename(self) -> str:
        """The name this file has in the bundle, and in a ``tiktoken`` cache: the cache key."""
        return cache_key(self.url)

    def as_json(self) -> dict[str, Any]:
        """The manifest form. Keys sorted by :func:`json.dumps`, so a rebuild diffs cleanly."""
        return {"sha256": self.sha256, "size": self.size, "url": self.url}


@dataclass(frozen=True, slots=True)
class VocabularyBundle:
    """A validated bundle on disk."""

    root: Path
    tiktoken_version: str
    encodings: Mapping[str, tuple[str, ...]]
    """Encoding name to the URLs it is built from, in the order ``tiktoken`` asks for them."""
    vocabularies: Mapping[str, BundledVocabulary]
    """URL to the file carried for it. Several encodings can name the same URL — ``p50k_base``
    and ``p50k_edit`` do — and the file is carried once."""

    @property
    def vocabulary_dir(self) -> Path:
        """Where the files are. Also a valid ``TIKTOKEN_CACHE_DIR``, read-only included."""
        return self.root / VOCABULARY_DIR_NAME

    @property
    def encoding_names(self) -> tuple[str, ...]:
        """What this bundle can seed, sorted."""
        return tuple(sorted(self.encodings))

    def path_for(self, url: str) -> Path:
        """The file carried for ``url``.

        Raises:
            VocabularyBundleError: The bundle does not carry it. A caller asking for a URL
                outside the bundle is asking the wrong bundle, which is worth a message rather
                than a ``KeyError``.
        """
        entry = self.vocabularies.get(url)
        if entry is None:
            msg = (
                f"the vocabulary bundle at {self.root} carries "
                f"{sorted(self.vocabularies)} and not {url!r}. Rebuild it against this "
                f"install's tiktoken with tools/build_vocabulary_bundle.py."
            )
            raise VocabularyBundleError(msg)
        return self.vocabulary_dir / entry.filename

    def verify(self) -> None:
        """Hash every file and compare it against the manifest.

        Not done on read. Reading a bundle happens on every pre-seed, and hashing to answer
        "is a bundle present" would make the cheap path expensive, while the copy in
        :meth:`seed` has to read every byte anyway and hashes them on the way past. This
        exists for the deliberate check — a build step confirming what it just wrote.

        Raises:
            VocabularyBundleError: A file is missing, truncated, or does not hash to what the
                manifest says.
        """
        for url in sorted(self.vocabularies):
            self._verified_bytes(url)

    def seed(self, encodings: Sequence[str], cache_dir: Path) -> tuple[str, ...]:
        """Copy what ``encodings`` need into ``cache_dir``. Returns the encodings it covered.

        The whole air-gapped path, and it is a file copy: no HTTP client, no digest to fetch,
        no registry lookup. Encodings the bundle does not carry are skipped rather than
        refused, because the caller — :func:`manicule.vocabularies.store.prefetch` — still has
        the network to try for them and is the one that decides whether their absence is fatal.

        Each file is hashed as it is copied and written under a temporary name that is only
        renamed into place once the hash matches. A half-written vocabulary under its real
        name is worse than an absent one: ``tiktoken`` would read it, find the digest wrong,
        delete it and fetch — which on this host means refuse.

        Args:
            encodings: Wanted encodings, in any order.
            cache_dir: ``tiktoken``'s cache. Created if it does not exist, because an
                air-gapped install's cache directory has never been written to.

        Returns:
            The encodings copied, sorted.

        Raises:
            VocabularyBundleError: A file is missing from the bundle directory, or its content
                does not match the manifest.
        """
        wanted = sorted(set(encodings) & set(self.encodings))
        if not wanted:
            return ()
        cache_dir.mkdir(parents=True, exist_ok=True)
        for url in sorted({url for name in wanted for url in self.encodings[name]}):
            entry = self.vocabularies[url]
            destination = cache_dir / entry.filename
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, prefix=f".{entry.filename}.", delete=False
            ) as handle:
                partial = Path(handle.name)
            try:
                partial.write_bytes(self._verified_bytes(url))
                # The bundle's own mode, not the temporary file's. `NamedTemporaryFile` creates
                # `0600`, which would make a cache seeded by one account unreadable to every
                # other — and a shared read-only cache is a deployment this bundle is meant to
                # support. What the operator gave the bundle is what the cache gets.
                shutil.copymode(self.path_for(url), partial)
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        return tuple(wanted)

    def _verified_bytes(self, url: str) -> bytes:
        """A file's content, checked against the manifest before it is handed back."""
        entry = self.vocabularies[url]
        path = self.path_for(url)
        try:
            content = path.read_bytes()
        except OSError as exc:
            msg = (
                f"the vocabulary bundle at {self.root} lists {url} as {entry.filename} and "
                f"that file cannot be read: {exc}. Rebuild the bundle with "
                f"tools/build_vocabulary_bundle.py."
            )
            raise VocabularyBundleError(msg) from exc
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != entry.size or digest != entry.sha256:
            msg = (
                f"{path} does not match the vocabulary bundle manifest: recorded "
                f"{entry.size} bytes / sha256 {entry.sha256}, found {len(content)} bytes / "
                f"sha256 {digest}. The bundle is corrupt or was edited after it was built; "
                f"rebuild it with tools/build_vocabulary_bundle.py."
            )
            raise VocabularyBundleError(msg)
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

    An operator seeing "no vocabulary and no network" needs to know which three places were
    checked, because the fix is to put a bundle in one of them.
    """
    if explicit is not None:
        return f"no usable vocabulary bundle at {explicit}"
    configured = os.environ.get(BUNDLE_DIR_ENV)
    if configured:
        return f"no usable vocabulary bundle at {configured} (from {BUNDLE_DIR_ENV})"
    return (
        f"no offline vocabulary bundle is installed: {BUNDLE_DIR_ENV} is unset and no "
        f"{BUNDLE_MODULE} distribution carries one"
    )


def read(root: Path) -> VocabularyBundle:
    """Read and validate the bundle at ``root``.

    Every check here answers a way a bundle can be wrong while looking right. A manifest from
    a different schema parses into the wrong shape. A manifest naming an encoding with no
    files is a bundle that reports itself present and seeds nothing. A truncated file is one
    ``tiktoken`` reads, finds wrong, deletes and re-fetches — which on the host this exists
    for means a refusal that blames the network.

    Sizes are checked here and hashes are not: a size check is a ``stat`` and catches the
    truncation case, while hashing is deferred to the copy in :meth:`VocabularyBundle.seed`,
    which reads every byte regardless.

    Raises:
        VocabularyBundleError: Anything above.
    """
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = (
            f"{root} is not a vocabulary bundle: it has no {MANIFEST_NAME}. Build one with "
            f"tools/build_vocabulary_bundle.py, or unset {BUNDLE_DIR_ENV} to fetch instead."
        )
        raise VocabularyBundleError(msg) from exc
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"the vocabulary bundle manifest {manifest_path} could not be read: {exc}"
        raise VocabularyBundleError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"the vocabulary bundle manifest {manifest_path} is not an object"
        raise VocabularyBundleError(msg)
    # Cast rather than annotate. A `dict` narrowed out of `json.loads` is `dict[Unknown,
    # Unknown]` under strict checking, and the Unknown spreads into every expression that
    # touches it; declaring the shape once here keeps the rest of this module fully checked
    # while being honest that the values are whatever the file contained. Every one of them
    # is validated below before it becomes anything.
    manifest = cast("Mapping[str, Any]", raw)

    schema = manifest.get("schema_version")
    if schema != SCHEMA_VERSION:
        msg = (
            f"the vocabulary bundle at {root} declares manifest schema {schema!r}, and this "
            f"manicule reads schema {SCHEMA_VERSION}. Rebuild the bundle with this version's "
            f"tools/build_vocabulary_bundle.py."
        )
        raise VocabularyBundleError(msg)

    bundle = VocabularyBundle(
        root=root,
        tiktoken_version=str(manifest.get("tiktoken_version", "")),
        encodings=_encodings(manifest, manifest_path),
        vocabularies=_vocabularies(manifest, manifest_path),
    )
    _check_complete(bundle)
    return bundle


def _encodings(manifest: Mapping[str, Any], path: Path) -> Mapping[str, tuple[str, ...]]:
    """The manifest's encoding table, or an error naming the file rather than a ``KeyError``."""
    encodings = manifest.get("encodings")
    if not isinstance(encodings, dict) or not encodings:
        msg = f"the vocabulary bundle manifest {path} lists no encodings"
        raise VocabularyBundleError(msg)
    entries = cast("Mapping[str, Any]", encodings)
    table: dict[str, tuple[str, ...]] = {}
    for name, value in entries.items():
        urls = cast("list[Any]", value) if isinstance(value, list) else []
        if not urls or not all(isinstance(url, str) for url in urls):
            msg = (
                f"the vocabulary bundle manifest {path} lists {name!r} with no vocabulary "
                f"files. An encoding that names nothing would be reported as carried and seed "
                f"nothing, which is worse than an absent bundle."
            )
            raise VocabularyBundleError(msg)
        table[str(name)] = tuple(str(url) for url in urls)
    return table


def _vocabularies(manifest: Mapping[str, Any], path: Path) -> Mapping[str, BundledVocabulary]:
    """The manifest's file table.

    Every conversion happens here, inside one ``try``. A size recorded as ``"large"`` is a
    corrupt manifest and must read as one; leaving the ``int()`` outside would let it escape
    as a bare ``ValueError`` from somewhere with no file name in it.
    """
    vocabularies = manifest.get("vocabularies")
    if not isinstance(vocabularies, dict) or not vocabularies:
        msg = f"the vocabulary bundle manifest {path} lists no vocabulary files"
        raise VocabularyBundleError(msg)
    entries = cast("Mapping[str, Any]", vocabularies)
    try:
        return {
            str(url): BundledVocabulary(
                url=str(url), sha256=str(entry["sha256"]), size=int(entry["size"])
            )
            for url, entry in entries.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"the vocabulary bundle manifest {path} has a malformed entry: {exc}"
        raise VocabularyBundleError(msg) from exc


def _check_complete(bundle: VocabularyBundle) -> None:
    """Every encoding's files described, present, and the length the manifest recorded."""
    wrong: list[str] = []
    for name in bundle.encoding_names:
        for url in bundle.encodings[name]:
            if url not in bundle.vocabularies:
                wrong.append(f"{name} needs {url} and the manifest describes no such file")
    for url in sorted(bundle.vocabularies):
        entry = bundle.vocabularies[url]
        path = bundle.vocabulary_dir / entry.filename
        try:
            size = path.stat().st_size
        except OSError:
            wrong.append(f"{entry.filename} ({url}: missing)")
            continue
        if size != entry.size:
            wrong.append(f"{entry.filename} ({size} bytes, expected {entry.size})")
    if wrong:
        msg = (
            f"the vocabulary bundle at {bundle.root} is incomplete: {'; '.join(wrong)}. It "
            f"was built wrong or damaged in transit; rebuild it with "
            f"tools/build_vocabulary_bundle.py."
        )
        raise VocabularyBundleError(msg)


def resolve(explicit: Path | None = None) -> VocabularyBundle | None:
    """The usable bundle for this install, or ``None`` when there is no bundle at all.

    The one call the pre-seed makes. ``None`` means "nothing to seed from, use the network";
    an exception means "a bundle is here and is wrong", which is never quietly downgraded to
    the first, because a bundle that is present and ignored is how an air-gapped host ends up
    waiting on a blob store it cannot reach.
    """
    root = locate(explicit)
    if root is None:
        return None
    return read(root)


def build(encodings: Sequence[str], destination: Path) -> VocabularyBundle:
    """Write a bundle for ``encodings`` into ``destination``. Returns what was written.

    Built from this machine's own ``tiktoken`` cache, which means the machine that builds a
    bundle is one that has already pre-seeded — so building is a copy, a hash and an
    assertion, never a download. A build cannot half-succeed against a flaky blob store and
    produce a bundle missing an encoding nobody notices until an air-gapped host refuses a
    question.

    **Which files an encoding needs is asked of ``tiktoken``, not computed**, by
    :func:`manicule.vocabularies.store.blobs_for` — and the digest recorded is the one
    ``tiktoken`` declares, re-asserted against the bytes actually on disk. A bundle whose
    manifest says something the installed library does not is caught here, where a rebuild is
    cheap, rather than on the host that cannot rebuild anything.

    Args:
        encodings: The encodings to bundle.
        destination: Created if absent. An existing bundle in it is overwritten.

    Returns:
        The bundle that was written, re-read from disk — so the return value is proof the
        manifest describes what is actually there rather than what was intended.

    Raises:
        ConfigError: An encoding is not one this install knows.
        VocabularyBundleError: A vocabulary is not in this machine's cache, or the bytes there
            do not match the digest ``tiktoken`` declares.
    """
    wanted = sorted(set(encodings))
    table: dict[str, tuple[str, ...]] = {}
    files: dict[str, BundledVocabulary] = {}

    vocabulary_dir = destination / VOCABULARY_DIR_NAME
    vocabulary_dir.mkdir(parents=True, exist_ok=True)
    for encoding in wanted:
        blobs = blobs_for(encoding)
        table[encoding] = tuple(blob.url for blob in blobs)
        for blob in blobs:
            if blob.url in files:
                continue
            files[blob.url] = _copy_into(blob, vocabulary_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tiktoken_version": _installed_tiktoken_version(),
        "encodings": {name: list(urls) for name, urls in table.items()},
        "vocabularies": {url: entry.as_json() for url, entry in files.items()},
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle = read(destination)
    bundle.verify()
    return bundle


def _copy_into(blob: Blob, vocabulary_dir: Path) -> BundledVocabulary:
    """Copy one vocabulary out of this machine's cache, asserting the digest on the way.

    The digest asserted is ``tiktoken``'s own declaration for the URL, not a checksum of
    whatever happened to be in the cache. A cache holding the wrong bytes under the right name
    is exactly what a bundle must not propagate: it would be copied to every air-gapped host
    and rejected there, by a library with no idea where the file came from.
    """
    source = cache_path(blob.url)
    try:
        content = source.read_bytes()
    except OSError as exc:
        msg = (
            f"{blob.name} is not in this machine's tiktoken cache at {source.parent}. A "
            f"bundle is built from a populated cache, never downloaded: pre-seed it first "
            f'with `python -c "from manicule import vocabularies; vocabularies.prefetch('
            f"['...'])\"` on a machine with network access."
        )
        raise VocabularyBundleError(msg) from exc
    digest = hashlib.sha256(content).hexdigest()
    if blob.sha256 is not None and digest != blob.sha256:
        msg = (
            f"{source} does not match the digest tiktoken declares for {blob.url}: expected "
            f"{blob.sha256}, found {digest}. The cache holds the wrong bytes under the right "
            f"name; delete that file and pre-seed again rather than bundling it."
        )
        raise VocabularyBundleError(msg)
    shutil.copyfile(source, vocabulary_dir / cache_key(blob.url))
    return BundledVocabulary(url=blob.url, sha256=digest, size=len(content))


def _installed_tiktoken_version() -> str:
    """The ``tiktoken`` release a bundle was built against, for provenance rather than a gate.

    Recorded so that a bundle can say what asked for these files, and reported by
    :func:`manicule.vocabularies.store.bundle_status`. Not checked on read: see this module's
    docstring for why a version mismatch is not, on its own, a reason to refuse.
    """
    from importlib.metadata import version  # noqa: PLC0415 - only needed when building

    return version("tiktoken")
