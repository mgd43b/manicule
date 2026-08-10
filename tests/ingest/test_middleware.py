"""The middleware contract, and every way of breaking it.

Each negative case here has a fake in :mod:`tests.ingest.fakes` that does the wrong thing on
purpose. Every one of them was verified by removing the corresponding check and watching this
file go red — a guard nobody has seen fire is a guard nobody knows works.
"""

from __future__ import annotations

import pytest

from manicule.core.errors import MiddlewareViolationError
from manicule.ingest.middleware import MiddlewareRunner, text_digest
from tests.fakes import make_chunks, make_document, make_raw
from tests.ingest import fakes


def _blocks() -> list[object]:
    from manicule.core.anchors import LineAnchor  # noqa: PLC0415 - local to the helper
    from manicule.core.content import BlockKind, ParsedBlock  # noqa: PLC0415

    return [
        ParsedBlock(kind=BlockKind.PROSE, text=text, anchor=LineAnchor(start=n, end=n))
        for n, text in enumerate(["alpha", "beta"], start=1)
    ]


async def test_a_hook_that_returns_a_new_value_has_an_effect() -> None:
    """The single most important line in the runner: the return value is assigned.

    A runner that folds handlers over a value and discards what they return works for hooks
    that mutate in place and silently ignores hooks that return a new object — so the system
    behaves correctly often enough to look correct, and fails only for plugins written in the
    idiomatic style the signature advertises.
    """
    runner = MiddlewareRunner([fakes.DeclaredRewriter()])
    document = make_document()
    chunks = make_chunks(document)

    returned = await runner.after_chunk(document, chunks)

    assert all(chunk.embed_text.endswith(" extra") for chunk in returned)
    assert not any(chunk.embed_text.endswith(" extra") for chunk in chunks), (
        "the caller's list must be untouched; a hook returns a new value rather than editing one"
    )


async def test_a_hook_returning_none_where_a_value_was_required_fails_the_document() -> None:
    """Substituting the original input would resurrect the bug the assignment fixes."""
    runner = MiddlewareRunner([fakes.DiscardingHook()])
    document = make_document()

    with pytest.raises(MiddlewareViolationError, match="discarding"):
        await runner.after_chunk(document, make_chunks(document))


async def test_a_hook_returning_the_wrong_type_fails_the_document() -> None:
    runner = MiddlewareRunner([fakes.WrongTypeHook()])

    with pytest.raises(MiddlewareViolationError, match="wrong-type"):
        await runner.before_parse(make_raw())


async def test_rewriting_chunk_text_is_refused() -> None:
    """The defect no parser suite can catch.

    Every parser passes its round-trip obligation before any middleware exists. A hook that
    rewrites ``text`` breaks the correspondence afterwards, leaving a corpus that is internally
    consistent and whose citations quote text the source document does not contain.
    """
    runner = MiddlewareRunner([fakes.TextRewriter()])
    document = make_document()

    with pytest.raises(MiddlewareViolationError, match=r"Chunk\.text"):
        await runner.after_chunk(document, make_chunks(document))


async def test_rewriting_block_text_is_refused() -> None:
    """The same corruption one hook earlier: block text becomes chunk text."""
    runner = MiddlewareRunner([fakes.BlockRewriter()])

    with pytest.raises(MiddlewareViolationError, match=r"ParsedBlock\.text"):
        await runner.after_parse(make_document(), _blocks())  # pyright: ignore[reportArgumentType]


async def test_rewriting_embed_text_without_declaring_it_is_refused() -> None:
    """It changes every vector while both fingerprint refusals pass.

    Neither fingerprint knows middleware exists, so an undeclared rewrite produces a corpus no
    fingerprint describes — and nothing downstream can detect it.
    """
    runner = MiddlewareRunner([fakes.UndeclaredRewriter()])
    document = make_document()

    with pytest.raises(MiddlewareViolationError, match="mutates_embedded_text"):
        await runner.after_chunk(document, make_chunks(document))


async def test_rewriting_embed_text_having_declared_it_is_allowed() -> None:
    """Redaction and context augmentation are exactly what ``after_chunk`` is for."""
    runner = MiddlewareRunner([fakes.DeclaredRewriter()])
    document = make_document()

    returned = await runner.after_chunk(document, make_chunks(document))

    assert all(chunk.embed_text.endswith(" extra") for chunk in returned)


async def test_only_declaring_middleware_reaches_the_chunk_fingerprint() -> None:
    """A middleware that changes no vector must not invalidate a corpus."""
    runner = MiddlewareRunner([fakes.PassThrough(), fakes.DeclaredRewriter()])

    assert runner.declarations({"declared": "2.1"}) == ("declared@2.1",)


async def test_the_declaration_set_does_not_depend_on_configuration_order() -> None:
    """Reordering middleware changes not one vector, so it must not read as a re-index."""
    first = MiddlewareRunner([fakes.DeclaredRewriter(), _SecondRewriter()])
    second = MiddlewareRunner([_SecondRewriter(), fakes.DeclaredRewriter()])

    assert first.declarations() == second.declarations()


async def test_a_middleware_with_no_known_version_is_still_recorded() -> None:
    """Its presence changes the space whether or not its version can be established."""
    runner = MiddlewareRunner([fakes.DeclaredRewriter()])

    assert runner.declarations() == ("declared@",)


def test_the_text_digest_cannot_be_fooled_by_moving_a_boundary() -> None:
    """``["ab", "c"]`` and ``["a", "bc"]`` must not hash alike, or the check is decoration."""
    assert text_digest(["ab", "c"]) != text_digest(["a", "bc"])


class _SecondRewriter(fakes.DeclaredRewriter):
    name = "another"
