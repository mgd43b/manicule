"""Where a ``tiktoken`` vocabulary comes from, and what happens when there is not one.

``tiktoken``'s wheel carries the BPE *implementation* and none of the *vocabularies*. Every
encoding it knows is a file on a Microsoft-hosted blob store, fetched on first use into a
cache directory and never mentioned again. So ``tiktoken.get_encoding("o200k_base")`` is a
network call wearing the clothes of a constructor, and manicule calls it on the query path —
the context fitter measures every assembled prompt with it.

Left alone that splits an air-gapped install's failure across two moments. Indexing succeeds,
which reads as a working install; the first *question* fails with a connection error naming a
blob storage host, which explains nothing and suggests less. It is the same hazard
:mod:`manicule.parsers.grammars` closes for grammars, arriving through a different library,
and it is closed the same way, in three moves:

1. **A pre-seed, never a lazy fetch.** :func:`prefetch` is the one function here that is
   allowed to reach the network. It is a step an operator runs and can watch fail, where a
   lazy fetch is a step that succeeds on the machine that built the image and not on the
   machine that runs it.
2. **An offline bundle first.** :mod:`manicule.vocabularies.bundle` is consulted before the
   network, so a host with no route to anything at all can still pre-seed — which is the
   difference between failing politely and working.
3. **The query path cannot fetch, at all.** :func:`load_encoding` shuts ``tiktoken``'s single
   door to the network for the duration of the call, so a vocabulary that was not pre-seeded
   is a refusal naming the encoding, the cache it looked in, and the bundle search path —
   never a download, and never a stack trace out of an HTTP client.

**Move 3 is what makes the other two more than documentation.** A fetch that is merely
*discouraged* still happens on the first machine whose cache was cleared by a temp sweep, and
it happens at the worst moment. Shutting the door means the failure is the same failure
everywhere, at a moment that names the fix.

**The door is one function.** ``tiktoken.load.read_file`` is the only place the library opens
a socket: ``get_encoding`` calls a constructor, the constructor calls ``load_tiktoken_bpe``,
that calls ``read_file_cached``, and *that* calls ``read_file`` only when the cache cannot
answer. Established by reading the installed package rather than assumed, and asserted by a
test that cuts the network at the socket layer in a subprocess — because a claim about which
function reaches the network is exactly the claim that a warm cache will let you make wrongly.

**Which files an encoding needs is asked, never derived.** The blob URLs and their expected
digests live in ``tiktoken_ext.openai_public`` as arguments inside constructor bodies; there
is no table to read. Hard-coding them in manicule would be a second definition of upstream's
data with nothing to keep the two in step. So :func:`blobs_for` runs the constructor with the
loader replaced by a recorder — the same principle as
:func:`manicule.parsers.grammar_bundle.build` asking the grammar pack which library answers
for which language rather than constructing the file name. The recorder serves anything
already in the cache and stops at the first file that is not, which is all a pre-seed needs to
know and costs no BPE table.

**The recorded identity does not change, and that is the point.**
:func:`manicule.chunking.tokens.tiktoken_tokenizer_id` records
``tiktoken/cl100k_base@0.13.0``, where the distribution version stands in for the vocabulary's
bytes. That was a *claim* while the bytes arrived from a URL nobody checked. It is now
checked: every byte this module puts in the cache is verified against the digest the installed
``tiktoken`` declares for it, so the release in the identifier really does name the bytes that
decided the boundaries. Adding a digest to the identifier would rewrite every chunk
fingerprint in every index to record something the version already implies.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from manicule.core.errors import ConfigError, ManiculeError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tiktoken import Encoding

__all__ = [
    "CACHE_DIR_ENV",
    "LEGACY_CACHE_DIR_ENV",
    "Blob",
    "VocabularyFetchError",
    "VocabularyUnavailableError",
    "blobs_for",
    "bundle_status",
    "cache_directory",
    "cache_key",
    "cache_path",
    "load_encoding",
    "missing_vocabularies",
    "prefetch",
    "required_encodings",
    "tiktoken_version",
]

CACHE_DIR_ENV: Final = "TIKTOKEN_CACHE_DIR"
"""``tiktoken``'s own cache directory variable, and the one a deployment should set.

Read here rather than set, for the same reason :mod:`manicule.parsers.grammars` reads the
pack's manifest variable: it is upstream's hook, and an installation that has already pointed
it somewhere — a container image, a shared read-only mount — meant it.

An install that sets neither this nor :data:`LEGACY_CACHE_DIR_ENV` gets a directory under the
system temporary directory, which is where ``tiktoken`` puts it and is **not** a durable
place. ``docs/deployment.md`` says to set this on any host that pre-seeds, because a temp
sweep that removes 5 MB of vocabulary turns a working air-gapped install back into a broken
one at the next question.
"""

LEGACY_CACHE_DIR_ENV: Final = "DATA_GYM_CACHE_DIR"
"""``tiktoken``'s older name for the same setting, still honoured by it and so honoured here.

Reproduced rather than ignored because this module answers "where would ``tiktoken`` read
this file from" and an answer that disagrees with ``tiktoken`` is worse than no answer: it
seeds a directory nothing reads, reports success, and leaves the fetch exactly where it was.
"""

CACHE_DIR_NAME: Final = "data-gym-cache"
"""``tiktoken``'s subdirectory under the system temporary directory, when nothing is set."""


class VocabularyUnavailableError(ManiculeError):
    """A vocabulary is needed and this install does not have it.

    Raised by :func:`load_encoding`, which never fetches. The message carries the three things
    an operator can act on — the encoding, the cache directory that was read, and where an
    offline bundle was looked for — because on the host where this fires, the network is
    almost never the actionable fact.
    """


class VocabularyFetchError(ManiculeError):
    """A pre-seed could not obtain a vocabulary.

    Separate from :class:`VocabularyUnavailableError` because they are read at different
    moments by people doing different things: this one is an install step failing, which is
    where such a failure belongs, and the other is a query refusing because that step never
    ran.
    """


@dataclass(frozen=True, slots=True)
class Blob:
    """One vocabulary file an encoding is built from, as ``tiktoken`` asks for it."""

    url: str
    """Where ``tiktoken`` would fetch it. Also its cache identity: the cache file name is a
    digest of this string, so the URL is the key even on a machine that will never open it."""

    sha256: str | None
    """The digest ``tiktoken`` declares for it, or ``None`` for an encoding that declares
    none. Recorded rather than computed, so a bundle carries upstream's expectation and not
    merely a checksum of whatever the builder happened to download."""

    @property
    def name(self) -> str:
        """The file's own name, for a message a human reads: ``o200k_base.tiktoken``."""
        return self.url.rsplit("/", 1)[-1]

    @property
    def key(self) -> str:
        """``tiktoken``'s cache file name for this blob."""
        return cache_key(self.url)


def required_encodings(context_encoding: str | None = None) -> tuple[str, ...]:
    """Every encoding this install will ask for, sorted.

    One definition, read by the pre-seed, the bundle builder, the container build, ``doctor``
    and CI, because copies of a list of encoding names is how a host ends up with the
    chunker's vocabulary and not the fitter's — an install that indexes and then cannot
    answer, which is the exact shape of the defect this package exists to close.

    The names come from the code that asks for them rather than from a constant beside them:
    the chunker's stand-in, the context fitter's configured encoding, and the generation
    budget's.

    Args:
        context_encoding: The configured ``rag.context.encoding``. Omitted, it is read from
            settings, which is right for a pre-seed script or an image build with nothing
            resolved in front of it — and wrong for a caller that has already resolved
            settings of its own, which would otherwise be told about a different installation.
    """
    from manicule.chunking.tokens import TIKTOKEN_ENCODING  # noqa: PLC0415 - lazy, see below
    from manicule.generation.budget import GENERATION_ENCODING  # noqa: PLC0415

    # Imported inside the function for the reason every import in this module is: a process
    # that only loads an encoding must not pay for the settings model, and nothing about a
    # pre-seed is on a hot path.
    if context_encoding is None:
        from manicule.config.settings import ContextSettings  # noqa: PLC0415

        context_encoding = ContextSettings().encoding
    return tuple(sorted({TIKTOKEN_ENCODING, context_encoding, GENERATION_ENCODING}))


def tiktoken_version() -> str:
    """The installed ``tiktoken`` release.

    Read from distribution metadata rather than from the module, so asking costs no import of
    the Rust extension. One definition, because three things say it and they must agree: the
    identity a provisional chunk records, the provenance a bundle carries, and what a status
    report prints. A vocabulary's bytes are verified against the digest *this* release
    declares, so the release is the honest name for what did the counting.
    """
    from importlib.metadata import version  # noqa: PLC0415 - see docstring

    return version("tiktoken")


def cache_key(url: str) -> str:
    """``tiktoken``'s cache file name for a blob URL: the SHA-1 of the URL, hex.

    Upstream's rule, reproduced because upstream exposes no function for it. A test asserts
    the reproduction against ``tiktoken`` itself — it seeds through ``tiktoken``'s own writer
    and checks the file lands where this says it will — so a release that changes the rule
    fails there rather than in a query on an air-gapped host.
    """
    return hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()


def cache_directory() -> Path:
    """Where ``tiktoken`` reads and writes vocabularies for this process.

    Upstream's resolution order, and the reason it is reproduced rather than read: there is no
    ``tiktoken.cache_dir()``. Getting it wrong is silent — a pre-seed writes 5 MB into a
    directory nothing consults and reports success — which is why it is a test's job to keep
    this honest rather than a comment's.
    """
    for variable in (CACHE_DIR_ENV, LEGACY_CACHE_DIR_ENV):
        configured = os.environ.get(variable)
        if configured:
            return Path(configured)
    return Path(tempfile.gettempdir()) / CACHE_DIR_NAME


def cache_path(url: str) -> Path:
    """Where the vocabulary at ``url`` sits in this process's cache, present or not."""
    return cache_directory() / cache_key(url)


_lock: Final = threading.RLock()
"""Held across every window in which ``tiktoken``'s loader is replaced.

Two windows exist — the probe in :func:`blobs_for` and the shut door in
:func:`load_encoding` — and both patch a module attribute that any thread could be calling.
Neither replacement can hand back wrong bytes: the probe serves the real cached file or
raises, and the door only ever raises. The lock is what stops the two from interleaving and
restoring each other's stand-in.
"""


class _Probed(Exception):  # noqa: N818 - control flow inside one function, never surfaced
    """The probe has learned what it came to learn and is stopping the constructor."""


def _constructor(encoding: str) -> Callable[[], Mapping[str, Any]]:
    """``tiktoken``'s own constructor for ``encoding``.

    Reached through :func:`tiktoken.list_encoding_names`, which is public and populates the
    registry's constructor table, so third-party ``tiktoken_ext`` plugins are found on the
    same terms as the built-in encodings.

    Raises:
        ConfigError: No installed plugin defines ``encoding``. Named as a configuration
            error and listed against what is available, because the way this is reached is
            ``rag.context.encoding`` holding a typo or a model name — and a model name is the
            specific mistake ``docs/retrieval.md`` §7.2 asks the fitter never to accept.
    """
    import tiktoken  # noqa: PLC0415 - a retrieval extra; importing manicule must not load it
    from tiktoken import registry  # noqa: PLC0415

    known = tiktoken.list_encoding_names()
    constructors = registry.ENCODING_CONSTRUCTORS or {}
    if encoding not in constructors:
        msg = (
            f"{encoding!r} is not a tiktoken encoding this install knows. Available: "
            f"{sorted(known)}. It must be an encoding name and never a model name — naming a "
            f"model would make an estimate look authoritative about a model that is not "
            f"being used."
        )
        raise ConfigError(msg)
    return constructors[encoding]


_enumerated: Final[dict[str, tuple[Blob, ...]]] = {}
"""Encodings whose full file list has been learned, in this process.

Only *complete* enumerations are kept — a probe that stopped at a gap has not seen what comes
after it. The list cannot change while a process runs: it is a property of the installed
``tiktoken``, not of what happens to be on disk. So this is a memo rather than a cache, and
what it buys is that the expensive shape of the probe happens at most once per encoding.
"""


def blobs_for(encoding: str) -> tuple[Blob, ...]:
    """Which vocabulary files ``encoding`` is built from, as far as this machine can see.

    Asked of ``tiktoken`` rather than derived from a table in manicule: the URLs and digests
    are arguments inside constructor bodies, so the only way to learn them that cannot go
    stale against an upstream release is to run the constructor and watch what it asks for.
    The loader is replaced for the duration with a recorder that hands back the real cached
    file when there is one and stops the constructor at the first file there is not.

    **What that costs, honestly.** When a file is missing the probe stops there, which is
    cheap and is the case a pre-seed acts on. When every file is present the constructor runs
    to completion and builds a 200 000-entry BPE table, because the only way to learn whether
    an encoding needs a *second* file is to let it ask for one. That full pass happens at most
    once per encoding per process: a complete answer is memoised in :data:`_enumerated`, and
    the file list is a property of the installed library rather than of the disk, so the memo
    cannot go stale. Nothing on the query path calls this.

    Returns:
        Every blob recorded, in the order the constructor asked for them. An enumeration that
        stopped at a gap is not memoised, so the next call — after a pre-seed has filled that
        gap — sees what comes after it.

    Raises:
        ConfigError: ``encoding`` is not one this install knows.
    """
    import tiktoken.load as loader  # noqa: PLC0415 - a retrieval extra, deliberately deferred

    build = _constructor(encoding)
    recorded: list[Blob] = []

    def record(blobpath: str, expected_hash: str | None = None) -> bytes:
        recorded.append(Blob(url=blobpath, sha256=expected_hash))
        path = cache_path(blobpath)
        if path.is_file():
            return path.read_bytes()
        raise _Probed

    with _lock:
        memoised = _enumerated.get(encoding)
        if memoised is not None:
            return memoised
        original = loader.read_file_cached
        loader.read_file_cached = record
        complete = False
        try:
            build()
            complete = True
        except Exception:  # noqa: BLE001 - see below
            # Every way a constructor can end is fine here, and none of them is this
            # function's business. :class:`_Probed` is the ordinary one — the answer arrived
            # and the constructor was stopped. The rest are the constructor failing on bytes
            # it was handed: a cache file that is the right length and the wrong content
            # parses into a ``ValueError`` halfway through a BPE table. Reporting that from
            # *here* would answer "which files does this encoding need" with a parse error,
            # and the file names — which is what was asked for — are already recorded.
            # Whoever goes on to load or bundle the encoding gets the real failure, from
            # ``tiktoken``'s digest check or from :func:`manicule.vocabularies.bundle.build`.
            complete = False
        finally:
            loader.read_file_cached = original
        blobs = tuple(recorded)
        if complete and blobs:
            _enumerated[encoding] = blobs
        return blobs


def _cached(blob: Blob) -> bool:
    """Whether this machine holds ``blob``, in the bytes ``tiktoken`` expects for it.

    The digest is checked and not merely the file name, because ``tiktoken``'s own loader
    checks it and *deletes the file and re-fetches* when it disagrees. A pre-seed that counted
    a wrong-bytes file as present would report success and leave the next query refusing, with
    the fix — copy the vocabulary again — sitting behind a message about a blob store. So a
    cache entry that will not be believed is reported as absent, which makes the pre-seed
    repair it from the bundle or the network like any other gap.
    """
    path = cache_path(blob.url)
    if not path.is_file():
        return False
    if blob.sha256 is None:
        return True
    return hashlib.sha256(path.read_bytes()).hexdigest() == blob.sha256


def missing_vocabularies(encodings: Sequence[str]) -> tuple[str, ...]:
    """Which of ``encodings`` this machine cannot build without a network. Sorted.

    What a pre-seed acts on and what a status report prints. No network access, and the cost
    is asymmetric in the useful direction: answering "this one is missing" stops at the first
    absent file, while answering "this one is here" reads every file the encoding needs —
    once to learn there are no more of them, and again to check the bytes are the ones
    ``tiktoken`` will accept.
    """
    absent: list[str] = []
    for encoding in sorted(set(encodings)):
        blobs = blobs_for(encoding)
        if not blobs or any(not _cached(blob) for blob in blobs):
            absent.append(encoding)
    return tuple(absent)


def load_encoding(encoding: str) -> Encoding:
    """Build ``encoding`` from what is already here, and never from the network.

    The one call the query path makes, and the reason this module exists. ``tiktoken``'s own
    ``get_encoding`` fetches when the cache cannot answer; this closes that door for the
    duration of the call, so an install that skipped the pre-seed gets a refusal that names
    the fix instead of a connection error that names a blob store.

    Encodings are cached inside ``tiktoken``, so the door is shut once per encoding per
    process in practice, and a second call costs a dictionary lookup.

    Raises:
        ConfigError: ``encoding`` is not one this install knows.
        VocabularyUnavailableError: A vocabulary file is not in the cache, or the copy there
            does not match the digest ``tiktoken`` expects. Both land here because both mean
            the same thing to whoever has to fix it: this machine cannot count tokens until
            the vocabulary is pre-seeded onto it.
    """
    import tiktoken  # noqa: PLC0415 - a retrieval extra; importing manicule must not load it
    import tiktoken.load as loader  # noqa: PLC0415

    _constructor(encoding)

    def refuse(blobpath: str) -> bytes:
        raise VocabularyUnavailableError(_unavailable(encoding, blobpath))

    with _lock:
        original = loader.read_file
        loader.read_file = refuse
        try:
            return tiktoken.get_encoding(encoding)
        finally:
            loader.read_file = original


def prefetch(encodings: Sequence[str], *, bundle_dir: Path | None = None) -> tuple[str, ...]:
    """Put every vocabulary in ``encodings`` on this machine. Returns what was seeded.

    The install step, and the only function here permitted to use the network. **The offline
    bundle is consulted first and the network only for what it did not supply**, which is what
    makes an air-gapped host work rather than merely fail politely — and is also the cheaper
    order on a host that does have a network, since a file copy beats a download.

    Args:
        encodings: Encoding names, in any order. Validated first, so a typo fails before
            anything is written.
        bundle_dir: An offline bundle to seed from. ``None`` looks in the places
            :func:`manicule.vocabularies.bundle.locate` describes; a bundle that is present
            and unusable is an error rather than a silent fall through to the network.

    Returns:
        The encodings actually seeded, sorted. Empty when everything was already here, which
        is the steady state and is what makes calling this on every start cheap.

    Raises:
        ConfigError: An encoding is not one this install knows.
        VocabularyBundleError: A bundle was found and is unusable.
        VocabularyFetchError: The vocabulary could not be obtained — no route to the blob
            store, a digest that did not match — **or it was obtained and is still not in the
            cache**, which is checked rather than assumed. See below.
    """
    from manicule.vocabularies import bundle as bundles  # noqa: PLC0415 - lazy, see docstring

    wanted = sorted(set(encodings))
    for encoding in wanted:
        _constructor(encoding)
    seeded = missing_vocabularies(wanted)
    if not seeded:
        return ()

    absent = seeded
    installed = bundles.resolve(bundle_dir)
    if installed is not None:
        installed.seed(absent, cache_directory())
        absent = missing_vocabularies(absent)
    if not absent:
        return seeded

    for encoding in absent:
        _fetch(encoding, bundle_dir)
    # `tiktoken` writes its cache on a best-effort basis: `read_file_cached` swallows every
    # OSError from the write when the cache directory was not named by an environment
    # variable. So a fetch can return a working encoding and leave the cache empty, and the
    # next process starts by downloading again — or, on the host this module exists for,
    # by refusing. Asking the cache afterwards costs a `stat` per file and turns that into an
    # error where it can still be acted on.
    still_missing = missing_vocabularies(absent)
    if still_missing:
        detail = (
            f"tiktoken supplied {list(still_missing)} and the cache at {cache_directory()} is "
            f"still without it — most often a cache directory that could not be written to"
        )
        raise VocabularyFetchError(_fetch_failure(still_missing, detail, bundle_dir))
    return seeded


def _fetch(encoding: str, bundle_dir: Path | None) -> None:
    """Let ``tiktoken`` download ``encoding``, and report a failure in terms of the fix.

    Under :data:`_lock` even though nothing here replaces the loader, because something else
    might be: a :func:`load_encoding` on another thread shuts the door for the duration of its
    call, and a download that happened to be in flight would find it shut and report a missing
    vocabulary it was in the middle of fetching. The lock is what keeps the one function
    allowed to use the network and the one function that forbids it from overlapping.
    """
    import tiktoken  # noqa: PLC0415 - a retrieval extra, deliberately deferred

    try:
        with _lock:
            tiktoken.get_encoding(encoding)
    except Exception as exc:
        # Deliberately broad. The ways this fails span three libraries and no common base:
        # `requests` raises its own hierarchy for a refused connection, a proxy and a TLS
        # failure; `tiktoken` raises `ValueError` for a digest mismatch and `ImportError`
        # when `requests` is absent; the filesystem raises `OSError`. Enumerating them means
        # the one that was left out escapes with a blob storage URL in it, which is the
        # message this whole module exists to replace.
        raise VocabularyFetchError(_fetch_failure([encoding], str(exc), bundle_dir)) from exc


def _unavailable(encoding: str, blobpath: str) -> str:
    """What a query says when a vocabulary was never pre-seeded onto this machine."""
    return (
        f"the {encoding!r} vocabulary is not on this machine: tiktoken needs "
        f"{blobpath.rsplit('/', 1)[-1]} and the cache at {cache_directory()} does not hold it "
        f"(or the copy there does not match the digest tiktoken expects). manicule never "
        f"downloads a vocabulary while answering a question, because that turns an install "
        f'step into an intermittent query failure. Pre-seed it — `python -c "from manicule '
        f"import vocabularies; vocabularies.prefetch(['{encoding}'])\"` on a host with "
        f"network access — or carry an offline bundle built by "
        f"tools/build_vocabulary_bundle.py. {_offline_summary(None)}. Set {CACHE_DIR_ENV} to "
        f"a durable directory if the cache above is under a temporary directory."
    )


def _fetch_failure(encodings: Sequence[str], detail: str, bundle_dir: Path | None) -> str:
    """One message for every way a pre-seed can fail, naming the bundle as well as the host.

    The bundle is quoted because this error is read most often on hosts that have no network:
    "connection refused" sends an operator to their firewall when the actionable fact is that
    there is no bundle, or that the bundle does not carry the encoding they configured.
    """
    return (
        f"could not obtain the {list(encodings)} vocabulary: {detail}. tiktoken ships no "
        f"vocabularies in its wheel and fetches them from openaipublic.blob.core.windows.net; "
        f"a host with no route there needs an offline bundle. {_offline_summary(bundle_dir)}. "
        f"Build one on a machine with network access with tools/build_vocabulary_bundle.py "
        f"and copy it to this host, or point {CACHE_DIR_ENV} at a cache that already has it."
    )


def _offline_summary(bundle_dir: Path | None) -> str:
    """:func:`bundle_status`, for a message that is already reporting a failure.

    A bundle that is present and broken raises, and raising *out of an error message* would
    replace the failure being reported with a different one — losing the connection refused,
    or the encoding that could not be built, that the reader needs. So the bundle problem is
    quoted into the message instead of thrown from it. Everywhere else, it is thrown.
    """
    from manicule.vocabularies import bundle as bundles  # noqa: PLC0415 - lazy, see docstring

    try:
        return bundle_status(bundle_dir)
    except bundles.VocabularyBundleError as exc:
        return f"the offline vocabulary bundle is unusable: {exc}"


def bundle_status(bundle_dir: Path | None = None) -> str:
    """One line describing what this install has to seed from offline.

    What every pre-seed failure quotes, and what a status command would print. An operator
    whose air-gapped host will not answer needs to know whether a bundle was found and which
    encodings it carries, because "no vocabulary and no network" has different fixes
    depending on the answer.

    Raises:
        VocabularyBundleError: A bundle exists and cannot be used. Reporting is not a reason
            to downgrade that to a shrug: a bundle that is present, wrong, and described as
            absent is how an operator ends up configuring a mirror they do not have.
    """
    from manicule.vocabularies import bundle as bundles  # noqa: PLC0415 - lazy, see docstring

    installed = bundles.resolve(bundle_dir)
    if installed is None:
        return bundles.describe_search_path(bundle_dir)
    return (
        f"offline vocabulary bundle at {installed.root}: built with tiktoken "
        f"{installed.tiktoken_version}, {len(installed.encodings)} encodings "
        f"({', '.join(installed.encoding_names)})"
    )
