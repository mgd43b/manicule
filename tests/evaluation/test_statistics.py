"""The arithmetic every verdict in this package rests on.

Checked against values that can be worked out by hand, because a statistics module nobody has
verified is the same hazard one layer down from the one this package exists to prevent: it
would make every p-value wrong in the same direction, and every report would still look fine.
"""

from __future__ import annotations

import math

import pytest

from manicule.evaluation.statistics import (
    binomial_tail,
    sign_test,
    smallest_detectable_sample,
    wilson_interval,
)


def test_the_tail_of_a_fair_coin_matches_the_hand_calculation() -> None:
    """``P(X >= 8 | 10, 0.5)`` is ``(45 + 10 + 1) / 1024``."""
    assert binomial_tail(8, 10, 0.5) == pytest.approx(56 / 1024)


def test_the_whole_distribution_sums_to_one() -> None:
    assert binomial_tail(0, 20, 0.3) == 1.0


def test_a_perfect_run_is_the_null_probability_raised_to_the_trials() -> None:
    assert binomial_tail(12, 12, 0.05) == pytest.approx(0.05**12)


def test_the_tail_survives_a_sample_size_that_overflows_the_direct_form() -> None:
    """``comb(2000, 1000)`` is past the largest representable double.

    Multiplying it by a probability raises ``OverflowError``, which is demonstrated here rather
    than described — so the log-space implementation is a fix for something that happens rather
    than a precaution. The size is not hypothetical: the sign test runs over every decided
    pairing ever recorded for a comparison, and a file of judgements is meant to grow.
    """

    def directly() -> float:
        """One term of the sum, computed the obvious way."""
        return math.comb(2000, 1000) * 0.5**1000

    with pytest.raises(OverflowError):
        directly()

    result = binomial_tail(1200, 2000, 0.5)

    assert not math.isnan(result)
    assert 0.0 < result < 1e-18


def test_impossible_and_certain_nulls_do_not_produce_a_number_by_accident() -> None:
    assert binomial_tail(1, 10, 0.0) == 0.0
    assert binomial_tail(10, 10, 1.0) == 1.0


@pytest.mark.parametrize(
    ("hits", "trials", "probability"),
    [(3, 2, 0.5), (1, 5, 1.5), (1, 5, -0.1)],
)
def test_impossible_inputs_raise_rather_than_returning_a_plausible_number(
    hits: int, trials: int, probability: float
) -> None:
    """Every one of these would otherwise return something a report would print."""
    with pytest.raises(ValueError, match=r"probability|possible"):
        binomial_tail(hits, trials, probability)


def test_a_small_sample_interval_still_contains_the_point_of_no_difference() -> None:
    """Seven of ten is the number this package exists to stop being quoted as a win."""
    low, high = wilson_interval(7, 10)

    assert low < 0.5 < high


def test_a_perfect_small_sample_does_not_report_a_zero_width_interval() -> None:
    """The normal approximation gives ``(1.0, 1.0)`` here — an exact claim from ten samples."""
    low, high = wilson_interval(10, 10)

    assert low < 1.0
    assert high == pytest.approx(1.0)


def test_no_trials_is_the_whole_range_rather_than_a_point() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_a_split_that_a_coin_would_produce_is_not_significant() -> None:
    assert sign_test(6, 4) > 0.05


def test_a_lopsided_split_is() -> None:
    assert sign_test(18, 2) < 0.01


def test_ties_are_dropped_rather_than_shared_between_the_sides() -> None:
    """Splitting them would manufacture evidence from judgements that expressed none."""
    assert sign_test(0, 0) == 1.0


def test_the_smallest_detectable_sample_is_the_point_where_a_perfect_run_clears_alpha() -> None:
    needed = smallest_detectable_sample(0.05, 0.01)

    assert binomial_tail(needed, needed, 0.05) <= 0.01
    assert binomial_tail(needed - 1, needed - 1, 0.05) > 0.01
