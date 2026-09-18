"""
shrinkage.py
=============
RB-Contact% Step 5.5: Empirical Bayes Beta-Binomial shrinkage of per-batter
RB-Contact% toward the league mean, weighted by swing count (paper eq. 3
analog). Self-contained -- no dependency on the modeling pipeline, so it's
unit-testable independently (see tests/test_shrinkage.py).
"""

import numpy as np


def fit_beta_prior_moments(rates: np.ndarray, weights: np.ndarray = None) -> tuple[float, float]:
    """
    Method-of-moments fit of Beta(alpha, beta) to a population of batter-level
    rates (e.g. per-batter RB-Contact%). `weights` (e.g. swing counts) let
    high-swing-count batters contribute more to estimating the population
    mean/variance, which is the more defensible choice when swing counts vary
    a lot -- but plain unweighted moments are used if weights is None.
    """
    rates = np.asarray(rates, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        mean = np.average(rates, weights=weights)
        var = np.average((rates - mean) ** 2, weights=weights)
    else:
        mean, var = rates.mean(), rates.var(ddof=1)

    if var <= 0 or var >= mean * (1 - mean):
        raise ValueError(
            f'Observed variance ({var:.6f}) is not consistent with a Beta '
            f'distribution at this mean ({mean:.4f}) -- need 0 < var < mean*(1-mean). '
            f'Population may be too homogeneous or too small to fit a prior.'
        )

    common = mean * (1 - mean) / var - 1
    alpha = mean * common
    beta = (1 - mean) * common
    return alpha, beta


def shrink_beta_binomial(p_hat: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Shrink each batter's rate `p_hat` (estimated from `n` swings) toward the
    league Beta(alpha, beta) prior mean alpha/(alpha+beta), weighted by `n`.

    Two closed-form identities this must satisfy exactly (see tests):
      - if p_hat == alpha/(alpha+beta) for all n, the output equals that same
        value regardless of n (shrinking a value already at the prior mean
        is a no-op)
      - if n == 0, the output equals alpha/(alpha+beta) regardless of p_hat
        (zero evidence -> pure prior)
    """
    p_hat = np.asarray(p_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    return (n * p_hat + alpha) / (n + alpha + beta)
