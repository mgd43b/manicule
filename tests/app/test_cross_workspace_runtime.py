"""A search spanning workspaces, end to end, over one real data directory holding several.

The shipped runtime, the shipped registry, SQLite and the embedded vector store, with each
workspace indexed through the real ingest pipeline. Only the embedder is a double, and it is a
precise one: every passage states its own cosine against the query (``relevance=0.93``), so the
order a merge produces is written down here rather than left to a hash — and a test that
disagreed with it would be disagreeing with arithmetic.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, override

import pytest

from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.errors import (
    FingerprintMismatchError,
    UnknownEntityError,
    VectorStoreStateError,
)
from manicule.core.ids import document_id
from manicule.plugins.registry import discover
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from manicule.core.embedding import Vector

SOURCE = "notes"
MODEL = "fake/relevance"
RELEVANCE = re.compile(r"relevance=([0-9.]+)")


class RelevanceEmbedder(HashEmbedder):
    """Embeds a passage at exactly the cosine it names against every query.

    A passage carrying ``relevance=w`` becomes ``(w, sqrt(1 - w²), 0, 0)`` and anything else —
    every query — becomes ``(1, 0, 0, 0)``, so the cosine between a query and a passage is ``w``.
    """

    def __init__(self, model_id: str = MODEL) -> None:
        super().__init__(dimension=4, model_id=model_id)

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            found = RELEVANCE.search(text)
            if found is None:
                vectors.append([1.0, 0.0, 0.0, 0.0])
                continue
            weight = float(found.group(1))
            vectors.append([weight, math.sqrt(1.0 - weight * weight), 0.0, 0.0])
        return vectors


def _runtime(
    environment: Path,
    workspace: str,
    *,
    embedder: Any | None = None,
    writer: bool = True,
    audit: bool = False,
    **rag: object,
) -> Runtime:
    """A runtime for ``workspace`` on the shared data directory, with a buildable pipeline."""
    from tests.ingest.fakes import BlockChunker  # noqa: PLC0415 - fakes, local to this harness

    found = discover()
    bound = found.registry.bind("test")
    selected = RelevanceEmbedder() if embedder is None else embedder
    bound.add(keys.EMBEDDER.named("local"), lambda _: selected)
    bound.add(keys.CHUNKER.named("block"), lambda _: BlockChunker())
    settings = Settings(
        data_dir=environment / "data",
        workspace=workspace,
        embedding={"provider": "local"},  # pyright: ignore[reportArgumentType]
        rag={"chunker": "block", **rag},  # pyright: ignore[reportArgumentType]
        security={"audit": {"enabled": audit}},  # pyright: ignore[reportArgumentType]
    )
    return Runtime(settings, discovery=found, writer=writer)


async def _index(
    environment: Path,
    workspace: str,
    passages: Mapping[str, str],
    *,
    embedder: Any | None = None,
) -> None:
    from tests.ingest.fakes import DictConnector  # noqa: PLC0415 - fakes, local to this harness

    source = DictConnector(dict(passages), name=SOURCE)
    for source_id in passages:
        source.media_types[source_id] = "text/plain"
    async with _runtime(environment, workspace, embedder=embedder) as opened:
        report = await (await opened.pipeline()).run(source)
        assert report.indexed == len(passages)


@pytest.fixture
async def corpus(manicule_environment: Path) -> Path:
    """Alpha and beta on one model, gamma on another, delta with no index at all."""
    await _index(
        manicule_environment,
        "alpha",
        {"alpha-high": "alpha orchard relevance=0.90", "alpha-low": "alpha orchard relevance=0.55"},
    )
    await _index(
        manicule_environment,
        "beta",
        {
            "beta-top": "beta orchard relevance=0.97",
            "beta-mid": "beta orchard relevance=0.93",
            "beta-low": "beta orchard relevance=0.60",
        },
    )
    await _index(
        manicule_environment,
        "gamma",
        {"gamma-any": "gamma orchard relevance=0.99"},
        embedder=RelevanceEmbedder(model_id="fake/another-model"),
    )
    async with _runtime(manicule_environment, "delta") as empty:
        await empty.documents()
    return manicule_environment


def _titles(result: Any) -> list[tuple[str, str]]:
    return [(hit.uri.removeprefix("memory://"), hit.workspace) for hit in result.hits]


async def test_two_workspaces_merge_on_cosine_and_every_hit_names_its_own(corpus: Path) -> None:
    """The settled rule, through every real component: cosine across workspaces, attributed."""
    async with _runtime(corpus, "alpha") as opened:
        result = await ApplicationService(opened).search(
            "orchard", workspaces=["alpha", "beta"], limit=5
        )

    assert _titles(result) == [
        ("beta-top", "beta"),
        ("beta-mid", "beta"),
        ("alpha-high", "alpha"),
        ("beta-low", "beta"),
        ("alpha-low", "alpha"),
    ]
    assert [round(hit.scores["dense"], 2) for hit in result.hits] == [0.97, 0.93, 0.9, 0.6, 0.55]
    assert result.workspaces == ("alpha", "beta")


async def test_the_ordinary_search_is_unchanged_by_a_spanning_one_beside_it(
    corpus: Path,
) -> None:
    """Naming only this workspace is the ordinary search, and so is naming none.

    Run after a spanning search of the same words, so that anything the two shared — a cached
    ranking, a bound leg, an opened handle — would show up here as a beta passage or as a
    ranking without the lexical leg. The cache is live across all three: a search's own
    query-log row does not move the key, so the spanning ranking is still cached when the
    ordinary search asks, and only the workspace set in the key keeps it from being served.
    """
    async with _runtime(corpus, "alpha") as opened:
        service = ApplicationService(opened)
        spanning = await service.search("orchard", workspaces=["alpha", "beta"])
        ordinary = await service.search("orchard")
        named = await service.search("orchard", workspaces=["alpha"])

    assert ordinary.cached is False, "the spanning ranking was served to a one-workspace search"
    assert named.cached is True, "naming only this workspace is the ordinary search, cached"
    assert {hit.workspace for hit in spanning.hits} == {"alpha", "beta"}
    assert _titles(ordinary) == [("alpha-high", "alpha"), ("alpha-low", "alpha")]
    assert set(ordinary.hits[0].scores) >= {"dense", "lexical", "rrf"}
    assert [hit.model_dump() for hit in named.hits] == [hit.model_dump() for hit in ordinary.hits]
    assert named.workspaces == ordinary.workspaces == ("alpha",)


async def test_a_repeated_spanning_search_is_served_from_the_cache(corpus: Path) -> None:
    """A spanning search writes a query-log row and an audit entry, and neither of them ranks.

    Audited on purpose: the audit entry is the record a spanning search writes and an ordinary
    one does not, so it is the write this path adds to the ones that used to move the key.
    """
    async with _runtime(corpus, "alpha", audit=True) as opened:
        service = ApplicationService(opened)
        first = await service.search("orchard", workspaces=["alpha", "beta"], limit=5)
        second = await service.search("orchard", workspaces=["alpha", "beta"], limit=5)
        telemetry = await opened.telemetry()
        audited = (await telemetry.audit_logs(event_type="search.cross_workspace"))[1]

    assert audited == 2, "both searches must have written their audit entry"
    assert first.cached is False
    assert second.cached is True, "a spanning search's own records invalidated its ranking"
    assert _titles(second) == _titles(first)


async def test_a_workspace_whose_best_passage_is_deleted_still_offers_its_next_best(
    corpus: Path,
) -> None:
    """The top-k trap, one row deep, over the real vector store and the real join.

    Beta's best passage is deleted, and its next-best still beats everything alpha holds. A
    fan-out that asked each store for exactly ``k`` and joined afterwards would get beta's
    deleted row, drop it, and report alpha's best as the best in either workspace.
    """
    async with _runtime(corpus, "beta") as beta:
        store = await beta.documents()
        await store.soft_delete_document(document_id("beta", SOURCE, "beta-top"))

    async with _runtime(corpus, "alpha", overrides={"candidates": 1, "final_top_k": 1}) as opened:
        result = await ApplicationService(opened).search(
            "orchard", workspaces=["alpha", "beta"], limit=1
        )

    assert _titles(result) == [("beta-mid", "beta")]


async def test_a_workspace_embedded_by_another_model_refuses_the_search(corpus: Path) -> None:
    """Two models' cosines are two scales; merging them would rank one model against another.

    Refused whole, naming the workspace and both models — never run without gamma, which would
    be a partial answer reported as a whole one.
    """
    async with _runtime(corpus, "alpha") as opened:
        with pytest.raises(FingerprintMismatchError) as refused:
            await ApplicationService(opened).search("orchard", workspaces=["alpha", "gamma"])

    message = str(refused.value)
    assert "'gamma'" in message
    assert MODEL in message
    assert "fake/another-model" in message


async def test_an_unknown_workspace_is_refused_naming_the_ones_that_exist(corpus: Path) -> None:
    """A misspelled workspace is a question the caller can correct, so the answer lists them."""
    async with _runtime(corpus, "alpha") as opened:
        with pytest.raises(UnknownEntityError) as refused:
            await ApplicationService(opened).search("orchard", workspaces=["alpha", "betta"])

    message = str(refused.value)
    assert "'betta'" in message
    assert "alpha, beta, delta, gamma" in message


async def test_a_workspace_with_no_index_is_refused_rather_than_prepared(corpus: Path) -> None:
    """Delta has a row and nothing embedded. Opening it for a search must not create an index."""
    async with _runtime(corpus, "alpha") as opened:
        with pytest.raises(VectorStoreStateError, match="'delta' has no index yet"):
            await ApplicationService(opened).search("orchard", workspaces=["alpha", "delta"])

    async with _runtime(corpus, "delta") as delta:
        assert (await (await delta.documents()).index_fingerprints()).embed is None


async def test_handles_are_kept_between_searches_and_released_with_the_runtime(
    corpus: Path,
) -> None:
    """One handle per workspace for the life of the process, and none after it."""
    opened = _runtime(corpus, "alpha")
    async with opened:
        first = await opened.open_workspaces(["beta"])
        second = await opened.open_workspaces(["beta"])
        assert first[0].leg.vectors is second[0].leg.vectors
        assert first[0].leg.docstore is second[0].leg.docstore
        serving = await opened.open_workspaces(["alpha"])
        assert serving[0].leg.vectors is await opened.prepared_vectors(), (
            "the serving workspace was opened a second time rather than reusing its own handle"
        )

    assert opened._workspaces._vectors == {}  # pyright: ignore[reportPrivateUsage]


async def test_a_workspace_reset_after_it_was_opened_is_reopened_rather_than_searched_stale(
    corpus: Path,
) -> None:
    """Another process resets beta and indexes it again while this one holds a handle on it.

    The handle is bound to the index identity it was opened against. Reused, it would refuse
    with a message about rebuilding the runtime — or search vectors that were reset — so each
    search reads the identity again and reopens a handle whose identity has moved: refused while
    beta has no index, and serving beta's new passages once it has one.
    """
    reader = _runtime(corpus, "alpha", writer=False)
    async with reader:
        service = ApplicationService(reader)
        before = await service.search("orchard", workspaces=["alpha", "beta"])
        assert "beta" in {hit.workspace for hit in before.hits}

        async with _runtime(corpus, "beta") as beta:
            await ApplicationService(beta).reset_index()
        with pytest.raises(VectorStoreStateError, match="'beta' has no index yet"):
            await service.search("orchard", workspaces=["alpha", "beta"])

        await _index(corpus, "beta", {"beta-new": "beta orchard relevance=0.99"})
        after = await service.search("orchard", workspaces=["alpha", "beta"], limit=1)

    assert _titles(after) == [("beta-new", "beta")]
