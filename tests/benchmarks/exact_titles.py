"""Round 7 of :mod:`tests.benchmarks.lexical_questions`: exact titles in the fused shortlist.

Issue #489. Round 5 found that the lexical index omits document titles. Round 6 recovered
title ranks by dropping the lexical leg for short lookups, but also changed headings and
made one title worse. This round measures document-title evidence directly.

Run from a frozen checkout on the same ``docs/`` export (``9ce34e6e``) as rounds 5 and 6::

    python -m tests.benchmarks.exact_titles --docs <export> --verdicts <bge.json>
    python -m tests.benchmarks.exact_titles --docs <export> --verdicts <qwen.json> \\
        --provider ollama --model qwen3-embedding:0.6b --prefix-scheme qwen3 --tokenizer <dir>
    python -m tests.benchmarks.exact_titles --choose <bge.json> <qwen.json>

**Candidates.** After shipped RRF, read live, visible documents that own a candidate, under
the query's scope. Match the whole query against the stored title after case folding and
collapsing whitespace. For a file URI whose basename equals the title, also match its stem.
If exactly one document in the pool matches and it is not already first, move its highest
fused chunk with a dense score to first. Keep every score and the other chunks' relative
order. A match present only in the lexical leg cannot promote. The candidates in tie order:

- ``dense-leader-title`` applies the rule only if the highest dense score in the pool
  belongs to the matched document, breaking dense ties as RRF does.
- ``shortlist-title`` applies it anywhere in the pool.

Both leave the shipped lexical ladder and fusion scores intact. The candidates read only
document records, never the expected answers. An exact match can also be a heading or an
identifier with a different intended answer; a promotion is evidence of a title lookup,
not proof of intent. If several documents outside the pool have the same title, this
instrument cannot tell; uniqueness here is over the pool.

**The round 7 rule.** Use round 6's questions, hybrid MRR@10 and dense baseline, measured
from the same indexes per model. Close at least half of the dense-over-hybrid title gap
without worsening any title. Keep the lower end of a 95% paired bootstrap interval above
-0.02 for every other family, including the two-word identifier set on its own. Apply the
same bound to each non-title family's subset on which the candidate moved a chunk to
first, when nonempty. Keep all unanswerable questions in ``none``. Apply the rule under
BGE-M3 and Qwen3-Embedding-0.6B independently. Choose a candidate that passes under both
by the best averaged non-title gain, ties within 0.005 going to the earlier candidate.
Without a qualifier, keep the shipped order. The title set is intentionally an easy
lookup and only 19 queries; independent heading, identifier and mail questions are the
regression checks. Report the extent of overlap with those sets before changing a default.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

from manicule.core.content import DocumentStatus
from manicule.core.readers import admits
from manicule.retrieval.fusion import RRFStage
from tests.benchmarks import query_shapes as qs
from tests.benchmarks.lexical_questions import TWO_WORDS

if TYPE_CHECKING:
    from collections.abc import Generator

    from manicule.core.content import Document
    from manicule.core.protocols import DocStore
    from manicule.core.retrieval import Candidate, Query


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _matches(text: str, document: Document) -> bool:
    title = _normalized(document.title)
    if text == title:
        return True
    uri = urlsplit(document.uri)
    if uri.scheme != "file":
        return False
    filename = PurePosixPath(unquote(uri.path))
    return _normalized(filename.name) == title and _normalized(filename.stem) == text


async def promote(
    store: DocStore,
    query: Query,
    ordered: list[Candidate],
    *,
    require_dense_leader: bool,
) -> list[Candidate]:
    """Move one exact-title match, keeping the fused scores and the rest of the order."""
    text = _normalized(query.text)
    dense = [candidate for candidate in ordered if "dense" in candidate.scores]
    if not text or not dense:
        return ordered
    ids = frozenset(candidate.chunk.document_id for candidate in ordered)
    if query.filter.document_ids:
        ids &= query.filter.document_ids
    if not ids:
        return ordered
    scope = query.filter.model_copy(update={"document_ids": ids})
    documents = await store.list_documents(scope, limit=len(ids))
    matches = {
        document.id
        for document in documents
        if document.id in ids
        and document.status is DocumentStatus.INDEXED
        and admits(scope.readers, document.access)
        and _matches(text, document)
    }
    if len(matches) != 1 or ordered[0].chunk.document_id in matches:
        return ordered
    if require_dense_leader:
        leader = min(dense, key=lambda c: (-c.scores["dense"], c.chunk.id))
        if leader.chunk.document_id not in matches:
            return ordered
    for i, candidate in enumerate(ordered):
        if candidate.chunk.document_id in matches and "dense" in candidate.scores:
            return [candidate, *ordered[:i], *ordered[i + 1 :]]
    return ordered


@contextmanager
def _promotion(
    store: DocStore, selected: set[str], *, require_dense_leader: bool
) -> Generator[None]:
    shipped = RRFStage.run

    async def run(stage: RRFStage, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        ordered = await shipped(stage, query, candidates)
        promoted = await promote(store, query, ordered, require_dense_leader=require_dense_leader)
        if promoted is not ordered:
            selected.add(query.text)
        return promoted

    with patch.object(RRFStage, "run", run):
        yield


def _incumbent(_store: DocStore, _selected: set[str]) -> nullcontext[None]:
    return nullcontext()


CANDIDATES: dict[str, qs.Substitution] = {
    "shipped": _incumbent,
    "dense-leader-title": partial(_promotion, require_dense_leader=True),
    "shortlist-title": partial(_promotion, require_dense_leader=False),
}

_WHOLE_FAMILIES = {name: family for name, family in qs.FAMILIES.items() if not family[2]} | {
    "identifiers, two words": ((TWO_WORDS,), (), False)
}
FAMILIES = _WHOLE_FAMILIES | {
    f"{name} that changed": (groups, forms, True)
    for name, (groups, forms, _) in _WHOLE_FAMILIES.items()
    if name != "titles"
}


def main(argv: Sequence[str] | None = None) -> None:
    qs.run_cli(argv, description=__doc__, candidates=CANDIDATES, round_number=7, families=FAMILIES)


if __name__ == "__main__":
    main()
