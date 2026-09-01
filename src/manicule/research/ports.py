"""What the loop needs from the rest of manicule, stated structurally.

One protocol, and it is deliberately the *narrow* one. The loop does not need a ``Retriever``;
it needs something that turns a query into a ranking, which is what
:class:`~manicule.app.ports.Retrieving` already is. Declaring it here rather than importing the
application's copy keeps ``manicule.research`` free of ``manicule.app``, so the dependency runs
one way: the application composes research, and research knows nothing about the application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from manicule.core.retrieval import Query
    from manicule.retrieval.retriever import RetrievalResult


@runtime_checkable
class Retrieving(Protocol):
    """One ranking for one query.

    The same shape the application service holds its retriever through, so a research run and a
    plain ``ask`` reach the corpus by the same call and cannot diverge in what they are allowed
    to see.
    """

    async def retrieve(self, query: Query) -> RetrievalResult: ...


__all__ = ["Retrieving"]
