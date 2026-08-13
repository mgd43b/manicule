"""Running middleware hooks, and refusing the three things a hook must not do.

**The single most important line in this module is** ``value = await hook.run(value)``. A
hook is a transform: it receives a value and returns the value the pipeline continues with. A
runner that folds handlers over a value and then discards the result behaves correctly for
middleware that mutates its argument in place — objects are passed by reference — and
silently ignores middleware that returns a new one. The system then works often enough to
look correct, which is the worst available outcome, because the failure surfaces only for the
subset of plugins written in the idiomatic style the signature advertises.

So the return value is assigned, and it is **validated rather than trusted**. A hook that
returns the wrong type, or ``None`` where a value was required, fails that document. Quietly
substituting the original input would resurrect exactly the bug above.

Three refusals, in the order they bite:

``Chunk.text`` and ``ParsedBlock.text`` are immutable
    Checked with one digest over the ordered text values before the first hook and after the
    last. ``Chunk.text`` is what a citation displays and what ``Parser.resolve`` must
    reproduce, and every parser is held to that round trip as a test obligation — so a
    middleware rewriting it breaks the correspondence *after every parser test has passed*,
    leaving a corpus that is internally consistent and whose citations quote text the source
    document does not contain. No fingerprint repairs that.

``embed_text`` is mutable and fingerprinted
    Rewriting it is legitimate; redaction and context augmentation both belong there. A
    middleware that does so declares ``mutates_embedded_text``, and this module computes the
    declaration set that :mod:`manicule.ingest.refusals` folds into the chunk fingerprint.

A hook that raises fails one document
    Not the batch, and not the hook. A hook that fails on one document is usually a document
    problem, and auto-disabling would make the corpus depend on ingest order.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from manicule.core.content import Chunk, Document, ParsedBlock, RawDocument
from manicule.core.errors import MiddlewareViolationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from manicule.core.protocols import Middleware


def text_digest(texts: Iterable[str]) -> bytes:
    """One digest over an ordered run of text values.

    NUL-separated so that ``["ab", "c"]`` and ``["a", "bc"]`` cannot hash alike, which is the
    difference between a check and the appearance of one. Cheap enough — a single hash over
    data already in memory, once per document per hook chain — that it runs always rather
    than under a debug flag. A check that runs only when somebody suspects a problem does not
    catch the problem nobody suspected.
    """
    return hashlib.sha256(b"\0".join(text.encode("utf-8") for text in texts)).digest()


def declarations(
    middleware: Sequence[Middleware], versions: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """``name@version`` for every middleware that declares it rewrites embedded text.

    Sorted, so that reordering configuration — a legitimate change that alters not one
    vector — does not read as a corpus-wide invalidation. Middleware carrying no version is
    recorded as ``name@`` rather than omitted: a middleware whose version cannot be
    established is still a middleware whose presence changes the space, and dropping it from
    the set would make its arrival invisible.
    """
    supplied = versions or {}
    return tuple(
        sorted(
            f"{hook.name}@{supplied.get(hook.name, '')}"
            for hook in middleware
            if hook.mutates_embedded_text
        )
    )


def chain(
    middleware: Sequence[Middleware], versions: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """``name@version`` for **every** configured middleware, declared or not.

    :func:`declarations` and this differ by one predicate and answer two questions, and reading
    them as the same question with a stricter filter is how the glossary fingerprint would have
    been given a guard that looks right and covers nothing.

    ``declarations`` names the hooks that rewrite ``embed_text``, because that is the field
    whose mutation changes vectors — the thing :class:`~manicule.core.fingerprints.ChunkFingerprint`
    describes. Detection never reads ``embed_text``. It reads a chunk's ``text``, which the
    runner's digests hold immutable, and its ``heading_path``, which **no check here covers**;
    and it reads whatever chunks the chunker produced, whose boundaries follow from block
    metadata that :meth:`MiddlewareRunner.after_parse` permits a hook to rewrite. Both of those
    routes are open to a hook that declares nothing at all, so the honest set for that
    fingerprint is the whole chain.

    The price is that configuring an unrelated hook makes every document's glossary lineage
    stale. That buys a sweep over stored chunks with no parser and no embedder in it, which is
    the cheapest repair manicule has — and it is the right direction to be wrong in, because the
    other one is a definition served from rules that no longer exist.
    """
    supplied = versions or {}
    return tuple(sorted(f"{hook.name}@{supplied.get(hook.name, '')}" for hook in middleware))


class MiddlewareRunner:
    """Runs the configured hooks in order, and holds them to the contract.

    Ordering comes from configuration, which is where a reader can see it. Registration order
    is not a contract, because it depends on entry-point enumeration — and a corpus whose
    chunk boundaries depend on which distribution was installed first is not reproducible.
    """

    def __init__(self, middleware: Sequence[Middleware]) -> None:
        self._middleware = tuple(middleware)

    @property
    def middleware(self) -> tuple[Middleware, ...]:
        return self._middleware

    def declarations(self, versions: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """The chunk-fingerprint contribution of this chain."""
        return declarations(self._middleware, versions)

    def chain(self, versions: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """The glossary-fingerprint contribution of this chain: every hook, not a subset."""
        return chain(self._middleware, versions)

    async def before_parse(self, raw: RawDocument) -> RawDocument | None:
        """Transform the fetched document, or return ``None`` if a hook dropped it.

        ``None`` is the one short-circuit in the whole middleware surface, and it is the
        hook's own return value rather than an exception because it is an ordinary outcome:
        a document excluded by configuration is skipped, not failed.
        """
        value = raw
        for hook in self._middleware:
            returned = await hook.before_parse(value)
            if returned is None:
                return None
            self._require(hook, "before_parse", returned, RawDocument)
            value = returned
        return value

    async def after_parse(
        self, document: Document, blocks: Sequence[ParsedBlock]
    ) -> list[ParsedBlock]:
        """Transform parsed blocks, refusing any rewrite of their text.

        Block text becomes chunk text, so rewriting it here is the same corruption as
        rewriting ``Chunk.text`` one hook later: the block's anchor still points at source
        text that no longer matches what the block claims.
        """
        value = list(blocks)
        before = text_digest(block.text for block in value)
        for hook in self._middleware:
            returned = await hook.after_parse(document, list(value))
            self._require_list(hook, "after_parse", returned, ParsedBlock)
            value = list(returned)
            if text_digest(block.text for block in value) != before:
                self._violation(hook, "after_parse", "ParsedBlock.text")
        return value

    async def after_chunk(self, document: Document, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Transform chunks, refusing any rewrite of their citable text.

        The undeclared-mutation check is here as well as in
        :func:`~manicule.testing.assert_middleware_contract` because the conformance suite
        runs against the chunks its author chose. A middleware whose rewrite fires only on
        documents containing an email address passes that suite and changes the vector space
        in production, and the pipeline is the only thing positioned to see it happen.
        """
        value = list(chunks)
        before = text_digest(chunk.text for chunk in value)
        embedded = {chunk.id: chunk.embed_text for chunk in value}
        for hook in self._middleware:
            returned = await hook.after_chunk(document, list(value))
            self._require_list(hook, "after_chunk", returned, Chunk)
            value = list(returned)
            if text_digest(chunk.text for chunk in value) != before:
                self._violation(hook, "after_chunk", "Chunk.text")
            if not hook.mutates_embedded_text and self._embedded_changed(embedded, value):
                msg = (
                    f"middleware {hook.name!r} rewrote Chunk.embed_text in after_chunk with "
                    f"mutates_embedded_text = False. That changes every vector while both "
                    f"fingerprint refusals pass, because neither fingerprint knows middleware "
                    f"exists — producing a corpus no fingerprint describes. Declare "
                    f"mutates_embedded_text = True so the pipeline folds it into the chunk "
                    f"fingerprint, and accept the re-index that follows."
                )
                raise MiddlewareViolationError(msg)
            embedded = {chunk.id: chunk.embed_text for chunk in value}
        return value

    async def after_store(self, document: Document) -> None:
        """Let hooks observe a committed document. The return value is deliberately nothing."""
        for hook in self._middleware:
            await hook.after_store(document)

    # --- refusals ------------------------------------------------------------------------

    @staticmethod
    def _embedded_changed(before: Mapping[str, str], after: Sequence[Chunk]) -> bool:
        return any(chunk.id in before and chunk.embed_text != before[chunk.id] for chunk in after)

    @staticmethod
    def _violation(hook: Middleware, where: str, field: str) -> None:
        msg = (
            f"middleware {hook.name!r} modified {field} in {where}. It is what a citation "
            f"displays and what resolving an anchor must reproduce, so changing it breaks the "
            f"round-trip guarantee after every parser test has already passed: the corpus "
            f"stays internally consistent while its citations quote text the source does not "
            f"contain. No fingerprint repairs that. Rewrite embed_text instead, and declare "
            f"mutates_embedded_text."
        )
        raise MiddlewareViolationError(msg)

    @staticmethod
    def _require(hook: Middleware, where: str, value: object, expected: type) -> None:
        if isinstance(value, expected):
            return
        msg = (
            f"middleware {hook.name!r} returned {type(value).__name__} from {where}, but the "
            f"hook is a transform and the pipeline continues with what it returns. Substituting "
            f"the original input instead would make a hook that returns a new value silently "
            f"do nothing, which is the failure this check exists to prevent. Return a "
            f"{expected.__name__}."
        )
        raise MiddlewareViolationError(msg)

    @classmethod
    def _require_list(cls, hook: Middleware, where: str, value: object, item: type) -> None:
        if not isinstance(value, list):
            cls._require(hook, where, value, list)
            return
        for element in cast("list[object]", value):
            cls._require(hook, where, element, item)


__all__ = ["MiddlewareRunner", "chain", "declarations", "text_digest"]
