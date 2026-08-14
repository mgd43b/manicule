"""The deterministic query router.

A pure function over the query text: no model call, no store access, and nothing but the text
decides the route. It runs before the cache, because trivial input should not consume a
generation call and a cache of trivial answers is a staleness bug with no upside. A utility
*handler* goes on to read a store — that is the answer being computed, not the route being
chosen.

**A route nothing returns is not a route.** Declaring four routes and implementing two gives a
type that documents a feature the function does not have. Every :class:`UtilityKind` this
router will emit is one a handler exists for, and it refuses to be constructed otherwise.

**Full match, never prefix, and tuned for precision.** A greeting route requires the entire
input to be a greeting, modulo surrounding punctuation and whitespace, under a short length
bound. The prefix version is not a small imprecision: a pattern anchored only at the start
routes *"yo-yo manufacturing tolerances"* away from the corpus, because ``-`` is a non-word
character and the word boundary matches, and *"thanks for the memory dump — what does it
say?"* to a canned reply. Both are ordinary queries against a technical corpus and both get an
answer that never touched the index. Anchoring at both ends deletes the entire class.

The governing rule, which also settles how much effort the pattern list deserves:

    **A missed greeting costs one retrieval, which is harmless. A false greeting costs a wrong
    answer to a real question, which is not. When in doubt, retrieve.**

That is also the answer to multilingualism. The corpus is multilingual by design and any
greeting list will be incomplete; the list is configuration, ships small, and being incomplete
costs only latency.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.errors import ConfigError
from manicule.retrieval.trace import Route

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from manicule.config.settings import RouterSettings


class UtilityKind(StrEnum):
    """A question about the index rather than about its contents."""

    DOCUMENT_COUNT = "document_count"
    DOCUMENT_LIST = "document_list"
    INDEX_STATUS = "index_status"


UTILITY_PHRASES: Final[Mapping[UtilityKind, tuple[str, ...]]] = {
    UtilityKind.DOCUMENT_COUNT: (
        "how many documents",
        "how many documents are indexed",
        "how many documents do you have",
        "how many docs",
        "document count",
    ),
    UtilityKind.DOCUMENT_LIST: (
        "list documents",
        "list all documents",
        "list the documents",
        "show documents",
        "what documents do you have",
    ),
    UtilityKind.INDEX_STATUS: (
        "index status",
        "sync status",
        "what is the index status",
        "is the index up to date",
    ),
}
"""Whole inputs that ask about the index.

Short and deliberately unclever. Every entry is a phrase nobody asks a document corpus, which
is the only property that matters: a phrase with a plausible reading as a real question does
not belong here at any level of coverage.
"""

_TRIM: Final = " \t\n\r.!?,;:'\"()[]{}…"


class Routing(BaseModel):
    """Where a query is going, and what decided it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Route = Route.RETRIEVE
    utility: UtilityKind | None = None
    matched: str = Field(
        default="", description="The pattern that matched, for the trace. Empty when none did."
    )

    @property
    def bypasses_retrieval(self) -> bool:
        """Whether the corpus will not be consulted.

        The one path where an answer legitimately has no sources, which is why it has to be
        *visibly* different rather than quietly identical: no citations, an explicit statement
        that the corpus was not consulted, confidence **absent** rather than 1.0 or 0.0, and no
        cache entry.
        """
        return self.route is not Route.RETRIEVE


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and strip surrounding punctuation.

    Everything a full match should tolerate and nothing it should not. Interior punctuation
    survives, so ``yo-yo manufacturing tolerances`` never becomes ``yo``.
    """
    collapsed = " ".join(text.split())
    return collapsed.strip(_TRIM).lower().strip()


class QueryRouter:
    """Decides whether a query reaches the corpus at all."""

    def __init__(
        self,
        settings: RouterSettings,
        *,
        available: Collection[UtilityKind] = (),
    ) -> None:
        """Build a router that emits only the routes something can answer.

        Args:
            settings: Pattern lists and the length bound.
            available: Utility kinds a handler exists for. Kinds outside this set are never
                emitted, so their phrases fall through to retrieval — which answers them
                badly rather than not at all, and is the correct failure for a route with
                nothing behind it.

        Raises:
            ConfigError: ``available`` names a kind this router has no phrases for, which
                would be a handler nothing can reach.
        """
        self._settings = settings
        self._available = frozenset(available)
        unreachable = sorted(kind.value for kind in self._available if kind not in UTILITY_PHRASES)
        if unreachable:
            msg = (
                f"utility handler(s) declared for {', '.join(unreachable)}, which the router "
                f"has no phrases for. A handler no route reaches is a feature nothing can "
                f"invoke; add phrases for it, or remove the handler."
            )
            raise ConfigError(msg)
        self._greetings = frozenset(normalize(phrase) for phrase in settings.greetings)
        self._utilities = {
            normalize(phrase): kind for kind in self._available for phrase in UTILITY_PHRASES[kind]
        }

    @property
    def utility_kinds(self) -> frozenset[UtilityKind]:
        """The kinds this router will emit."""
        return self._available

    def route(self, text: str) -> Routing:
        """Where ``text`` goes. Pure, deterministic and store-free."""
        if not self._settings.enabled:
            return Routing()
        if len(text) > self._settings.max_chars:
            # A greeting is short. A sentence that begins with one is a question, and the
            # length bound is what stops the second being read as the first.
            return Routing()

        normalized = normalize(text)
        if not normalized:
            return Routing()
        utility = self._utilities.get(normalized)
        if utility is not None:
            return Routing(route=Route.UTILITY, utility=utility, matched=normalized)
        if normalized in self._greetings:
            return Routing(route=Route.GREETING, matched=normalized)
        return Routing()


__all__ = ["UTILITY_PHRASES", "QueryRouter", "Routing", "UtilityKind", "normalize"]
