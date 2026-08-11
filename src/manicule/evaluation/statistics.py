"""The small-sample statistics the harness refuses to report without.

Three functions, and each exists because a number without one of them reads as evidence when
it is not:

- :func:`binomial_tail` turns "it got 6 of 20 right" into "chance would do that 3% of the
  time", which is the only form in which a hit rate says anything.
- :func:`wilson_interval` turns "it won 7 of 10" into an interval that visibly contains 0.5,
  which is what stops a ten-query run being quoted as a win.
- :func:`sign_test` says whether a preference split is distinguishable from a coin.

Pure Python and no dependency, deliberately. ``numpy`` and ``scipy`` are implementation
dependencies that core does not carry, and an evaluation harness that cannot be imported
without a numerical stack is one that gets skipped. Everything here is exact to floating point
over the sample sizes a hand-judged query set reaches.

The computations run in log space rather than multiplying binomial coefficients by powers. A
binomial coefficient is an arbitrary-precision integer and a probability is a float, so the
direct form has to convert one to the other, and past about 1030 trials ``comb(n, n // 2)``
exceeds the largest representable double and the multiplication raises ``OverflowError``. That
is not a hypothetical size: the sign test in a report runs over every decided pairing ever
recorded for a comparison, and a file of judgements is meant to grow — so the direct form is an
implementation that works right up until the evaluation set is large enough to be worth having.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054
"""The standard normal quantile for a two-sided 95% interval.

A literal rather than a call into a statistics package, because it is a constant of the normal
distribution and this module exists so that no numerical stack is needed to read a report.
"""


def binomial_tail(hits: int, trials: int, probability: float) -> float:
    """``P(X >= hits)`` for ``X ~ Binomial(trials, probability)``.

    The upper tail rather than the lower one, because every question this harness asks is "did
    it do *better* than chance", and a two-sided answer to a one-sided question halves the
    evidence for no reason.

    Args:
        hits: Successes observed. Zero or fewer makes the tail the whole distribution, so the
            answer is 1.0 — which is correct and is also the honest reading of "we observed
            nothing surprising".
        trials: How many attempts. Zero trials means no evidence, so 1.0 again.
        probability: The null hypothesis, in ``[0, 1]``.

    Raises:
        ValueError: ``probability`` is outside ``[0, 1]``, or ``hits`` exceeds ``trials``.
            Both are caller bugs that would otherwise return a plausible number.
    """
    if not 0.0 <= probability <= 1.0:
        msg = f"probability must be in [0, 1], got {probability!r}"
        raise ValueError(msg)
    if trials < 0:
        msg = f"trials must not be negative, got {trials}"
        raise ValueError(msg)
    if hits > trials:
        msg = f"observed {hits} hits in {trials} trials, which is not possible"
        raise ValueError(msg)
    if hits <= 0 or trials == 0:
        return 1.0
    if probability <= 0.0:
        # Nothing can succeed under this null, so any success at all is infinitely surprising.
        # Reported as 0.0 rather than raising: a corpus large enough to make chance underflow
        # is a good corpus, not a broken one.
        return 0.0
    if probability >= 1.0:
        return 1.0

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_factorials = math.lgamma(trials + 1)
    total = 0.0
    for successes in range(hits, trials + 1):
        log_term = (
            log_factorials
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + successes * log_p
            + (trials - successes) * log_q
        )
        total += math.exp(log_term)
    return min(1.0, total)


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> tuple[float, float]:
    """A confidence interval for a proportion, by Wilson's score method.

    Wilson rather than the textbook normal approximation, and the reason is exactly the regime
    this harness lives in: at ``7/10`` the normal interval is ``(0.42, 0.98)`` and at ``10/10``
    it is ``(1.0, 1.0)`` — an interval of zero width around a proportion measured from ten
    samples. Wilson gives ``(0.72, 1.0)`` there, which is a claim a reader can weigh.

    Returns:
        Lower and upper bounds, clamped to ``[0, 1]``. With no trials the interval is the whole
        range, because that is what "no evidence" looks like.
    """
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def sign_test(left: int, right: int) -> float:
    """Two-sided ``p`` for a preference split, ties already excluded.

    The standard sign test: under "the two systems are indistinguishable" each decided pairing
    is a fair coin, so the question is how often a coin produces a split at least this lopsided
    in *either* direction. Two-sided here and one-sided in :func:`binomial_tail`, because this
    question genuinely has two interesting answers — the challenger can lose — while "beats
    chance" has one.

    Ties and "neither" are dropped rather than split between the sides. Splitting them would
    manufacture evidence from judgements that deliberately expressed none.
    """
    decided = left + right
    if decided == 0:
        return 1.0
    return min(1.0, 2.0 * binomial_tail(max(left, right), decided, 0.5))


def smallest_detectable_sample(probability: float, alpha: float) -> int:
    """The fewest trials at which a *perfect* run could reach significance.

    The question this answers is the one an underpowered probe cannot: if a system got every
    single item right, would that be enough to reject chance? Below this many trials the answer
    is no, and a probe run there reports "indistinguishable from chance" for a flawless system
    as readily as for a useless one. A check whose failing verdict is unconditional is not a
    check, so the probe refuses to run rather than emitting it.

    Returns:
        The smallest ``n`` with ``P(X = n | n, probability) <= alpha``. ``1`` when even a single
        trial suffices, which happens only for a very large corpus.
    """
    if not 0.0 < probability < 1.0:
        msg = f"probability must be strictly between 0 and 1, got {probability!r}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be strictly between 0 and 1, got {alpha!r}"
        raise ValueError(msg)
    # P(X = n) = probability ** n, so n >= log(alpha) / log(probability). Computed rather than
    # searched, because a loop here would be doing arithmetic the closed form already did.
    return max(1, math.ceil(math.log(alpha) / math.log(probability)))


__all__ = [
    "Z_95",
    "binomial_tail",
    "sign_test",
    "smallest_detectable_sample",
    "wilson_interval",
]
