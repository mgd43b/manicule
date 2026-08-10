"""Resolving the profile a query runs under, and how deep the pipeline goes for it.

``Query.profile`` names one of three cost/quality settings and ``rag.overrides`` adjusts
individual fields of it. Both are read here rather than in each stage, so that a stage cannot
resolve them slightly differently from its neighbour and produce a pipeline whose depth
depends on which stage you ask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.config.profiles import ProfileConfig, profile_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manicule.core.retrieval import Query


class Profiles:
    """The profile resolver every stage in one pipeline shares."""

    def __init__(self, overrides: Mapping[str, object] | None = None) -> None:
        self._overrides = dict(overrides or {})

    @property
    def overrides(self) -> Mapping[str, object]:
        """The per-field overrides in force, for the run's recorded identity."""
        return dict(self._overrides)

    def for_query(self, query: Query) -> ProfileConfig:
        """The settings this query runs under.

        Overrides start from the named profile rather than from a separate set of defaults, so
        overriding one field cannot silently move another.
        """
        return profile_config(query.profile, self._overrides)


def retrieval_depth(profile: ProfileConfig, query: Query) -> int:
    """How many candidates each leg fetches and each stage works over.

    Two settings mean "how many candidates come out" and neither document reconciled them, so:
    ``Query.limit`` is what a search call returns to a person, and ``final_top_k`` is what an
    ask call puts in the model's context. They have different consumers and both are honoured,
    which means the pipeline has to run at least as deep as the larger of them.

    ``final_top_k`` does not appear here because a profile guarantees it is no larger than
    ``candidates`` — a configuration returning more candidates than it fetched is refused where
    the profile is built, not worked around here.
    """
    return max(profile.candidates, query.limit)


__all__ = ["ProfileConfig", "Profiles", "retrieval_depth"]
