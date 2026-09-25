"""
suite_mda_metric.py
====================
Defines the MDA metric that survived Step 1, attaches bootstrap CIs, and runs
the m2 recalibration check.

The metric under test (used by Steps 2 and 3)
----------------------------------------------
`mda_resid` = per-batter residual of MDA_context after partialling out
contact% AND the pitch-mix descriptors (mean release_speed_c, mean
|plate_x_bat_flip|, mean plate_z, fastball share, breaking share). NOT raw
MDA, and NOT the variant that also removes the model's own mean prediction --
that last control is over-control for a calibrated model (see below).

The m2 recalibration question
------------------------------
"Did we settle for m3, or actually need it?" A between-batter recalibration of
the context model is

    expected_i = a + s * mean_pred_i        (s = calibration slope)
    MDA_recal_i = mean_actual_i - expected_i

which is exactly the residual of MDA regressed on mean_pred -- i.e.
recalibrating IS partialling out `mean_pred_offset`. So the recalibrated
metric is already the `contact_plus_pitchmix` row from Step 1. This module
verifies that equivalence numerically rather than asserting it, then compares
all three variants under one uniform, maximally-conservative control so the
m2-vs-m3 choice is apples-to-apples.

Bootstrap CIs are cluster (batter-level) resamples: draw batters with
replacement, recompute discrimination. That is the right resampling unit --
the question is whether the between-batter signal is real, so the batter is
the sampling unit, not the swing.

Outputs
-------
  mda_metric_per_batter.csv    per-batter metric values for every variant
  mda_discrimination_ci.csv    discrimination + 95% CI, both control regimes
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import VARIANT_ORDER, OUT_DIR, TRAIN_SEASON, EVAL_SEASON
from suite_data import load_train_and_eval
from followup_utils import discrimination
from suite_partial_mda_v2 import (build_swings, aggregate, fit_ols,
                                  exact_sampling_var, PREDICTORS,
                                  PITCHMIX_ONLY, RESID_COL)

warnings.filterwarnings('ignore')

N_BOOT_CI = 2000
SEED = 42

REGIMES = {
    # The metric under test: legitimate controls only.
    'pitchmix': PITCHMIX_ONLY,
    # Uniform maximally-conservative control, for apples-to-apples variant
    # comparison (equivalent to recalibrating the context model).
    'pitchmix_plus_pred': [p for p, _ in PREDICTORS],
}


def build_metric(variant: str, eval_df: pd.DataFrame) -> dict:
    df = build_swings(variant, eval_df)
    bat = aggregate(df)
    y = bat['mda'].to_numpy(float)

    out = {'bat': bat, 'df': df, 'regimes': {}}
    for name, cols in REGIMES.items():
        beta, e, r2, _ = fit_ols(y, bat[cols].to_numpy(float))
        var_e = exact_sampling_var(df, bat, cols, beta)
        out['regimes'][name] = dict(resid=e, var=var_e, r2=r2, beta=beta,
                                    cols=cols)
    return out


def boot_ci(values: np.ndarray, var: np.ndarray, n_boot: int,
            seed: int) -> tuple[float, float]:
    """Cluster bootstrap over batters."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(values) & np.isfinite(var)
    v, s = values[ok], var[ok]
    n = len(v)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = discrimination(v[idx], s[idx])
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def recalibration_equivalence(variant: str, m: dict) -> dict:
    """
    Recalibrate explicitly, then check it reproduces the partialled residual.
    expected_i = a + s*mean_pred_i from regressing mean ACTUAL on mean PRED.
    """
    bat, df = m['bat'], m['df']
    actual = df.groupby('batter')['miss_distance_t'].mean().reindex(bat.index)
    pred = bat['mean_pred_offset'].to_numpy(float)
    s, a = np.polyfit(pred, actual.to_numpy(float), 1)
    mda_recal = actual.to_numpy(float) - (a + s * pred)

    # Partialling MDA on mean_pred alone should give the same thing.
    beta_p, e_p, _, _ = fit_ols(bat['mda'].to_numpy(float),
                                pred.reshape(-1, 1))
    return dict(
        variant=variant, calib_slope=float(s),
        corr_recal_vs_partialled=float(np.corrcoef(mda_recal, e_p)[0, 1]),
        max_abs_diff=float(np.max(np.abs(mda_recal - e_p))),
        sd_recal=float(np.std(mda_recal, ddof=1)),
    )


def main():
    ap = argparse.ArgumentParser(description='MDA metric definition + CIs')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    ap.add_argument('--n-boot', type=int, default=N_BOOT_CI)
    args = ap.parse_args()

    _, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON, verbose=False)

    rows, per_batter, equiv = [], [], []
    for v in args.variants:
        m = build_metric(v, eval_df)
        bat = m['bat']

        for regime, r in m['regimes'].items():
            d = discrimination(r['resid'], r['var'])
            lo, hi = boot_ci(r['resid'], r['var'], args.n_boot, SEED)
            rows.append(dict(variant=v, regime=regime, r2_partialled=r['r2'],
                             discrimination=d, ci_lo=lo, ci_hi=hi,
                             n_batters=int(len(bat)),
                             resid_sd=float(np.std(r['resid'], ddof=1))))

        pb = pd.DataFrame({
            'batter': bat.index, 'variant': v, 'n_swings': bat['n'].values,
            'mda_raw': bat['mda'].values,
            'mda_resid': m['regimes']['pitchmix']['resid'],
            'mda_resid_var': m['regimes']['pitchmix']['var'],
            'mda_resid_conservative': m['regimes']['pitchmix_plus_pred']['resid'],
            'contact_pct': bat['contact_pct'].values,
            'mean_pred_offset': bat['mean_pred_offset'].values,
        })
        per_batter.append(pb)
        equiv.append(recalibration_equivalence(v, m))

    ci = pd.DataFrame(rows)
    pd.concat(per_batter, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, 'mda_metric_per_batter.csv'), index=False)
    ci.to_csv(os.path.join(OUT_DIR, 'mda_discrimination_ci.csv'), index=False)

    pd.set_option('display.width', 220)
    print('=== Discrimination with 95% cluster-bootstrap CI ===\n')
    print(ci.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print('\n=== Recalibration == partialling out mean_pred (verification) ===\n')
    print(pd.DataFrame(equiv).to_string(index=False,
                                        float_format=lambda v: f'{v:,.6f}'))

    print('\n=== m2 vs m3 under the uniform conservative control ===')
    c = ci[ci['regime'] == 'pitchmix_plus_pred'].set_index('variant')
    if {'m2_merf_battrack', 'm3_rf_nore'} <= set(c.index):
        a, b = c.loc['m2_merf_battrack'], c.loc['m3_rf_nore']
        overlap = not (a['ci_hi'] < b['ci_lo'] or b['ci_hi'] < a['ci_lo'])
        print(f"  m2 {a['discrimination']:.4f} [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]")
        print(f"  m3 {b['discrimination']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]")
        print(f"  CIs overlap: {overlap} -> "
              f"{'indistinguishable' if overlap else 'distinguishable'}")
    print(f"\n-> {os.path.join(OUT_DIR, 'mda_discrimination_ci.csv')}")


if __name__ == '__main__':
    main()
