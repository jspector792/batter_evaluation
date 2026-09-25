"""
suite_partial_mda.py
=====================
Robustness check: is MDA adding anything raw contact% does not already have?

The worry is structural, not statistical. With the new target,
`miss_distance_t == 0` is definitionally `is_contact`, so for batter i

    mean(miss_distance_t)_i = (1 - contact_i) * mean(miss_distance | whiff)_i

which makes MDA_i mechanically a function of the batter's contact rate times
their average miss severity, less the model's expectation. A high
discrimination for MDA could therefore be contact% wearing a different hat.

The test
--------
1. Regress MDA_context on raw contact% across the cohort. Report R^2.
2. Take the residuals -- MDA with contact rate partialled out.
3. Compute Franks discrimination on that residual alone.

If the residual's discrimination collapses toward what a pure-noise variable
scores, MDA carries no independent signal.

Getting the sampling variance right is the whole ballgame
---------------------------------------------------------
Discrimination needs the sampling variance of each batter's own value. For the
partialled residual e_i = MDA_i - (a + b*contact_i), that is NOT simply MDA's
sampling variance: MDA_i and contact_i are computed from the SAME swings, so
their sampling errors are correlated -- strongly, given the definitional link.
Ignoring the covariance term would bias the answer. Two estimates are
reported:

  analytic   Var(e_i) = Var(MDA_i) + b^2 Var(contact_i)
                        - 2b Cov(MDA_i, contact_i)
             with the within-batter covariance estimated per batter as
             Cov_swing(offset_residual, is_contact) / n_i

  bootstrap  resample each batter's swings with replacement, recompute MDA*
             and contact*, form e* against the FIXED full-data coefficients,
             and take the spread of e* as the noise. Makes no distributional
             assumption and picks up the correlation automatically.

A pure-noise reference is also computed: per batter, a synthetic metric with
the same n and no true between-batter signal. Its discrimination is the
"collapsed to nothing" yardstick the comparison needs.

Both metrics are computed on the SAME swing subset (those with a non-null
offset residual) so the covariance is well defined and the regression is
apples-to-apples. That makes contact% here differ very slightly from the
dirB table, which uses all swings.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import VARIANT_ORDER, OUT_DIR, EVAL_SEASON
from followup_utils import discrimination

warnings.filterwarnings('ignore')

MIN_SWINGS_METRIC = 100
N_BOOT = 400
SEED = 42


def per_batter(df: pd.DataFrame, resid_col: str) -> pd.DataFrame:
    """
    Per-batter MDA, contact%, their sampling variances, and the within-batter
    sampling covariance between them. Restricted to swings where the residual
    exists so both means are over an identical set of swings.
    """
    sub = df.dropna(subset=[resid_col, 'is_contact']).copy()
    sub['_c'] = sub['is_contact'].astype(float)

    g = sub.groupby('batter')
    out = pd.DataFrame({
        'mda': g[resid_col].mean(),
        'mda_sd': g[resid_col].std(ddof=1),
        'contact': g['_c'].mean(),
        'n': g.size(),
    })
    # Within-batter swing-level covariance between the residual and the
    # contact indicator; the covariance of the two MEANS is this over n.
    cov = g.apply(lambda d: np.cov(d[resid_col].to_numpy(),
                                   d['_c'].to_numpy(), ddof=1)[0, 1])
    out['cov_swing'] = cov
    out = out[out['n'] >= MIN_SWINGS_METRIC].copy()

    out['var_mda'] = out['mda_sd'] ** 2 / out['n']
    out['var_contact'] = out['contact'] * (1 - out['contact']) / out['n']
    out['cov_means'] = out['cov_swing'] / out['n']
    return out, sub


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Return (intercept, slope, r2) for y ~ x."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(beta[0]), float(beta[1]), float(1 - ss_res / ss_tot)


def bootstrap_resid_noise(sub: pd.DataFrame, bat: pd.DataFrame, resid_col: str,
                          a: float, b: float, n_boot: int,
                          seed: int) -> np.ndarray:
    """
    Per-batter sampling variance of the partialled residual, by resampling
    that batter's own swings. Coefficients are held at their full-data values:
    we want the noise in e_i given the partialling, not noise in the
    partialling itself.
    """
    rng = np.random.default_rng(seed)
    out = np.full(len(bat), np.nan)
    groups = {k: (v[resid_col].to_numpy(), v['_c'].to_numpy())
              for k, v in sub.groupby('batter')}
    for i, batter in enumerate(bat.index):
        r, c = groups[batter]
        n = len(r)
        idx = rng.integers(0, n, size=(n_boot, n))
        mda_star = r[idx].mean(axis=1)
        contact_star = c[idx].mean(axis=1)
        e_star = mda_star - (a + b * contact_star)
        out[i] = e_star.var(ddof=1)
    return out


def noise_reference(bat: pd.DataFrame, seed: int) -> float:
    """
    Discrimination of a metric with NO between-batter signal: each batter's
    value is drawn purely from their own sampling distribution around one
    common mean. This is the number the residual would match if MDA carried
    nothing beyond contact%.
    """
    rng = np.random.default_rng(seed)
    sd = float(np.sqrt(bat['var_mda'].mean()))
    vals = rng.normal(0.0, np.sqrt(bat['var_mda'].to_numpy()))
    return discrimination(vals, bat['var_mda'].to_numpy())


def run_variant(variant: str, n_boot: int) -> dict:
    path = os.path.join(OUT_DIR, f'dirC_swing_residuals_{variant}.parquet')
    df = pd.read_parquet(path)
    resid_col = 'offset_residual_context'

    bat, sub = per_batter(df, resid_col)
    sub = sub[sub['batter'].isin(bat.index)]

    y = bat['mda'].to_numpy()
    x = bat['contact'].to_numpy()
    a, b, r2 = ols(y, x)
    e = y - (a + b * x)

    # Discrimination of MDA itself, for the before/after comparison.
    d_mda = discrimination(y, bat['var_mda'].to_numpy())

    # Analytic sampling variance of the partialled residual.
    var_e_analytic = (bat['var_mda'] + b ** 2 * bat['var_contact']
                      - 2 * b * bat['cov_means']).to_numpy()
    d_resid_analytic = discrimination(e, var_e_analytic)

    # Bootstrap sampling variance of the partialled residual.
    var_e_boot = bootstrap_resid_noise(sub, bat, resid_col, a, b, n_boot, SEED)
    d_resid_boot = discrimination(e, var_e_boot)

    # Naive version: wrongly reusing MDA's own sampling variance, ignoring the
    # covariance. Reported only to show how much that shortcut misleads.
    d_resid_naive = discrimination(e, bat['var_mda'].to_numpy())

    d_null = noise_reference(bat, SEED)

    return dict(
        variant=variant, n_batters=int(len(bat)),
        r2_mda_on_contact=r2, slope=b,
        pearson_r=float(np.corrcoef(y, x)[0, 1]),
        disc_MDA=d_mda,
        disc_resid_analytic=d_resid_analytic,
        disc_resid_bootstrap=d_resid_boot,
        disc_resid_naive_no_cov=d_resid_naive,
        disc_pure_noise_reference=d_null,
        var_resid_obs=float(np.var(e, ddof=1)),
        mean_var_resid_analytic=float(np.nanmean(var_e_analytic)),
        mean_var_resid_boot=float(np.nanmean(var_e_boot)),
        resid_sd_over_mda_sd=float(np.std(e, ddof=1) / np.std(y, ddof=1)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()

    rows = [run_variant(v, args.n_boot) for v in args.variants]
    out = pd.DataFrame(rows)

    path = os.path.join(OUT_DIR, 'dirB_mda_partialled.csv')
    out.to_csv(path, index=False)

    pd.set_option('display.width', 250)
    print(f'MDA_context regressed on raw contact%, {EVAL_SEASON} cohort '
          f'(>= {MIN_SWINGS_METRIC} swings)\n')
    print(out.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
