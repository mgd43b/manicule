"""PII redaction at the generation boundary.

Redaction happens **on the way out to a model**, never at ingest. Ingest-time redaction is
incoherent while original bytes are retained — the unredacted source is in the blob store
regardless, so it protects nothing while permanently degrading retrieval — and re-parsing
through a changed redactor would churn chunk ids and vectors on every pattern edit.

**It is a projection applied to a copy.** The verification chain — ``Chunk.text`` ↔ ``Anchor``
↔ retained original bytes — never sees a redacted string. If verification ran against what
was *sent*, every redacted passage would fail its round trip and the feature would delete
every citation it touched.

A consequence worth stating rather than discovering: **a citation can point at a span
containing text the model never saw.** The prompt said ``[REDACTED]``; the citation resolves
to the original address. That is the feature working. Redaction controls *what leaves the
machine*, not what is hidden from the person who indexed the corpus and already has read
access to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from manicule.config.settings import RedactionMethod, RedactionSettings
from manicule.core.errors import ConfigError, RedactionError


@dataclass(frozen=True, slots=True)
class Detector:
    """One named, versioned detector.

    Named rather than expressed as a raw regex in configuration, so detectors can be tested,
    improved and reasoned about — and so a config file is a policy rather than a program.
    :attr:`version` exists because improving a pattern changes what leaves the machine, and a
    trace that records only the name cannot say which pattern was in force.
    """

    name: str
    version: int
    pattern: re.Pattern[str]


def _detector(name: str, version: int, pattern: str) -> Detector:
    return Detector(name, version, re.compile(pattern))


BUILTIN_DETECTORS: Mapping[str, Detector] = {
    detector.name: detector
    for detector in (
        # Bounded quantifiers throughout. Every pattern here runs over up to 32k tokens of
        # context on every query, and a nested or unbounded quantifier is how a redactor
        # becomes a denial-of-service surface against the machine it protects.
        _detector("email", 1, r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}"),
        _detector(
            "phone",
            1,
            r"(?<![\w.])\+?\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]?\d{2,4}){1,3}(?![\w.])",
        ),
        _detector("credit-card", 1, r"(?<!\d)(?:\d{4}[ -]?){3}\d{1,4}(?!\d)"),
        _detector(
            "ip-address",
            1,
            r"(?<![\w.:])(?:\d{1,3}(?:\.\d{1,3}){3}|(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4})(?![\w.:])",
        ),
    )
}
"""The supported surface.

**Recall-oriented, and they will fire on things that are not personal data** — a version
string that looks like a phone number, an internal identifier that looks like a card. That is
why the feature is off by default and why enabling it is a decision with a stated cost rather
than a free precaution.
"""


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Redacted text and what fired, by detector name.

    **Counts, never values.** Recording what a detector matched would turn a diagnostic into
    the leak the detector existed to prevent.
    """

    text: str
    counts: Mapping[str, int] = field(default_factory=dict[str, int])

    @property
    def changed(self) -> bool:
        return any(self.counts.values())


def resolve_detectors(settings: RedactionSettings) -> tuple[Detector, ...]:
    """Every detector this configuration selects, built-in and custom.

    Raises:
        ConfigError: A named detector does not exist, or a custom pattern does not compile.
            Both are refusals rather than silent omissions: a detector that is not running
            makes redaction quietly weaker than the configuration says it is.
    """
    detectors: list[Detector] = []
    unknown: list[str] = []
    for name in settings.patterns:
        found = BUILTIN_DETECTORS.get(name.strip().lower())
        if found is None:
            unknown.append(name)
        else:
            detectors.append(found)
    if unknown:
        available = ", ".join(sorted(BUILTIN_DETECTORS))
        msg = (
            f"security.data_policy.auto_redact.patterns names {', '.join(unknown)}, which is "
            f"not a built-in detector. Available: {available}. Add a regex to "
            f"custom_patterns if you meant one of your own."
        )
        raise ConfigError(msg)
    for index, pattern in enumerate(settings.custom_patterns):
        try:
            detectors.append(Detector(f"custom[{index}]", 1, re.compile(pattern)))
        except re.error as exc:
            msg = (
                f"security.data_policy.auto_redact.custom_patterns[{index}] is {pattern!r}, "
                f"which is not a valid regular expression: {exc}. A pattern that does not "
                f"compile cannot redact anything, and dropping it silently would leave "
                f"redaction weaker than this configuration claims."
            )
            raise ConfigError(msg) from exc
    return tuple(detectors)


class Redactor:
    """Applies the configured detectors to text leaving for a model.

    Constructed once. Construction is where an impossible configuration is refused, before
    any query has been answered.
    """

    def __init__(self, settings: RedactionSettings) -> None:
        self._settings = settings
        self._detectors = resolve_detectors(settings)
        self._salt = self._require_salt()

    def _require_salt(self) -> bytes:
        """The per-installation secret ``hash`` needs, or a refusal naming how to make one.

        ``hash`` exists for one reason and it is worth stating: it preserves co-reference, so
        the same address is the same token in every passage and the model can still tell that
        two mentions are one person. That only holds if the digest is keyed. **An unsalted
        digest of an email address is reversible by anyone with a word list**, and a truncated
        one collides — so sending a hash instead of the value would be privacy theatre that
        costs answer quality and buys nothing.

        Generated rather than defaulted, because a default salt is no salt at all: every
        installation would produce the same digest for the same address.
        """
        if self._settings.method is not RedactionMethod.HASH:
            return b""
        secret = self._settings.hash_salt
        if secret is None or not secret.get_secret_value():
            msg = (
                "security.data_policy.auto_redact.method is 'hash' but hash_salt is unset. An "
                "unsalted digest of an email address is reversible with a word list, so it "
                "would send the value in a costume rather than protect it. Generate one that "
                'never leaves this machine — python -c "import secrets; '
                'print(secrets.token_hex(32))" — and set '
                "security.data_policy.auto_redact.hash_salt."
            )
            raise ConfigError(msg)
        return secret.get_secret_value().encode("utf-8")

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return self._detectors

    @property
    def enabled(self) -> bool:
        return self._settings.enabled and bool(self._detectors)

    def redact(self, text: str) -> RedactionResult:
        """Apply every detector. Synchronous; see :meth:`redact_all` for the deadline."""
        if not self.enabled or not text:
            return RedactionResult(text)
        counts: dict[str, int] = {}
        redacted = text
        for detector in self._detectors:
            hits = 0

            def substitute(match: re.Match[str]) -> str:
                nonlocal hits
                hits += 1
                return self._replacement(match.group(0))

            redacted = detector.pattern.sub(substitute, redacted)
            if hits:
                counts[detector.name] = counts.get(detector.name, 0) + hits
        return RedactionResult(redacted, counts)

    def _replacement(self, value: str) -> str:
        method = self._settings.method
        if method is RedactionMethod.REMOVE:
            return ""
        if method is RedactionMethod.HASH:
            digest = hmac.new(self._salt, value.encode("utf-8"), hashlib.sha256).hexdigest()
            return f"{self._settings.replacement}:{digest[:16]}"
        return self._settings.replacement

    async def redact_all(self, texts: Sequence[str]) -> tuple[list[str], Mapping[str, int]]:
        """Redact several strings under one wall-clock deadline.

        **A regex over 32k tokens of context, with operator-supplied patterns, is a
        denial-of-service surface**, and Python's ``re`` cannot be interrupted — so the work
        runs in a worker thread and the deadline is enforced from outside it.

        **Exceeding it fails the query.** The fail-safe direction is refuse-to-send: there is
        no path in this design where a timeout, an exception or a mistake results in
        unredacted text reaching a remote model.

        The honest limit, because a security control that is quietly approximate is worse
        than one that is openly so: the *query* fails at the deadline, but the thread running
        the pattern is not killed, because nothing can kill it. A catastrophically
        backtracking custom pattern therefore costs a worker thread until it finishes. That is
        the operator's risk, and it is why custom patterns are separated from the supported
        built-ins.

        **The thread is a daemon, and that detail is load-bearing** rather than a style
        choice. :func:`asyncio.to_thread` uses the default executor, whose threads are
        non-daemon and are joined at interpreter exit — so a runaway pattern would not merely
        leak a thread, it would hang the process's shutdown, turning a bad regex into a server
        that will not stop. A daemon thread costs the same leak and lets the process exit.

        Raises:
            RedactionError: The deadline expired, or a pattern raised.
        """
        if not self.enabled:
            return list(texts), {}
        try:
            return await asyncio.wait_for(
                _in_daemon_thread(lambda: self._redact_all_blocking(texts)),
                self._settings.timeout_s,
            )
        except TimeoutError as exc:
            msg = (
                f"redaction exceeded security.data_policy.auto_redact.timeout_s "
                f"({self._settings.timeout_s}s) over {len(texts)} strings, so nothing was "
                f"sent. Python's regex engine cannot be interrupted, so a pattern that "
                f"backtracks catastrophically shows up here. Simplify or remove the custom "
                f"patterns, or raise the deadline if the corpus is simply large."
            )
            raise RedactionError(msg) from exc
        except re.error as exc:
            msg = f"a redaction pattern failed while running, so nothing was sent: {exc}"
            raise RedactionError(msg) from exc

    def _redact_all_blocking(self, texts: Sequence[str]) -> tuple[list[str], Mapping[str, int]]:
        results = [self.redact(text) for text in texts]
        counts: dict[str, int] = {}
        for result in results:
            for name, hits in result.counts.items():
                counts[name] = counts.get(name, 0) + hits
        return [result.text for result in results], counts


async def _in_daemon_thread[T](work: Callable[[], T]) -> T:
    """Run blocking work on a thread the interpreter will not wait for at exit.

    See :meth:`Redactor.redact_all` for why the default executor is the wrong home for a
    regex that may not terminate.
    """
    loop = asyncio.get_running_loop()
    finished: asyncio.Future[T] = loop.create_future()

    def settle() -> None:
        try:
            outcome = work()
        except BaseException as exc:  # noqa: BLE001 - relayed to the awaiting side verbatim
            loop.call_soon_threadsafe(_set_exception, finished, exc)
        else:
            loop.call_soon_threadsafe(_set_result, finished, outcome)

    threading.Thread(target=settle, name="manicule-redaction", daemon=True).start()
    return await finished


def _set_result[T](future: asyncio.Future[T], value: T) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception[T](future: asyncio.Future[T], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


def detector_names(detectors: Iterable[Detector]) -> tuple[str, ...]:
    """Names and versions, for a trace that has to say which patterns were in force."""
    return tuple(f"{detector.name}@{detector.version}" for detector in detectors)


__all__ = [
    "BUILTIN_DETECTORS",
    "Detector",
    "RedactionResult",
    "Redactor",
    "detector_names",
    "resolve_detectors",
]
