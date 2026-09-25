"""
suite_mda_validity.py
======================
Steps 2 and 3 of the MDA follow-up, run on the metric that survived Step 1.

Metric under test
-----------------
`mda_resid` -- per-batter MDA_context with contact% AND the pitch-mix
descriptors partialled out (suite_mda_metric.REGIMES['pitchmix']). Raw MDA and
the over-controlled variant are NOT used here.

Step 2 -- redundancy against public stats
------------------------------------------
Rates computed directly from the pitch data, so all 520 cohort batters are
covered and the definitions match the modelling population exactly:

    whiff_pct         1 - contact% over swings
    zone_contact_pct  contact% on in-zone swings (zone <= 9)
    chase_pct         swing rate on out-of-zone pitches. Computed over ALL
                      pitches seen (no bat_speed filter) to match the public
                      definition -- it is a plate-discipline stat, not a
                      swing-quality one.

From data/batter_stats_{season}.csv:

    hard_hit_pct      ev_ev95percent, a raw rate (655 batters)
    xwoba             xst_est_woba (658)
    squared_up_pctl   pct_squared_up_rate -- a PERCENTILE RANK (1-100), not a
                      rate, and only 205 batters. Percentile ranks are a
                      monotone transform of the underlying rate, so Spearman
                      is meaningful for them and Pearson is not. Both are
                      reported; read Spearman for the *_pctl rows.

Note `whiff_pct` is definitionally 1 - contact%, and contact% is one of the
things partialled out of mda_resid. A near-zero correlation there is a
construction check, not evidence of novelty.

Step 3 -- out-of-sample predictive validity
--------------------------------------------
Does an early-sample mda_resid add anything to an early-sample raw rate when
predicting the full season? Per the standing rule, the comparison is against
the EB-SHRUNK raw estimate, never the unshrunk one.

At each sample fraction, the partialling regression is refit WITHIN the early
sample (520 batters, 6 predictors -- ample), so no full-season information
leaks into the early metric.

Three estimators per target:

    raw_direct      EB-shrunk early raw rate used as the prediction, unfitted.
                    The dirA-style baseline, kept for continuity.
    raw_fitted      5-fold CV OLS of the target on the shrunk early rate.
    raw_plus_mda    5-fold CV OLS on the shrunk early rate AND the shrunk
                    early mda_resid.

`raw_fitted` vs `raw_plus_mda` is the actual test: both are fitted and
cross-validated, so the only difference is the extra predictor. Comparing an
unfitted baseline against a fitted augmented model would hand the augmented
model a free rescaling advantage.

Outputs
-------
  redundancy_correlations.csv
  mda_stabilization.csv
  plots/mda_stabilization.png
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (OUT_DIR, PLOT_DIR, DATA_ROOT, TRAIN_SEASON,
                          EVAL_SEASON, ALL_SWINGS, BUNT_DESC, CONTACT)
from suite_data import load_train_and_eval
from suite_partial_mda_v2 import (build_swings, aggregate, fit_ols,
                                  exact_sampling_var, PITCHMIX_ONLY, RESID_COL)
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial
from followup_utils import CHRON_KEYS, rmse

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY = '#2563EB', '#DC2626', '#16A34A', '#6B7280'

SAMPLE_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_FOLDS = 5
SEED = 42
DEFAULT_VARIANT = 'm3_rf_nore'


# ──────────────────────────────────────────────────────────────────────────
# Shrinkage helpers
# ──────────────────────────────────────────────────────────────────────────

def shrink_rate(values, n):
    try:
        a, b = fit_beta_prior_moments(np.asarray(values, float),
                                      weights=np.asarray(n, float))
    except ValueError:
        return np.asarray(values, float)
    return shrink_beta_binomial(np.asarray(values, float),
                                np.asarray(n, float), a, b)


def shrink_normal(values, sampling_var):
    """
    Normal empirical-Bayes shrinkage toward the grand mean, for a continuous
    metric. Needed because the beta-binomial form only applies to rates, and
    leaving mda_resid unshrunk while shrinking the rate would bias the
    comparison against mda_resid at small samples.
    """
    v = np.asarray(values, float)
    s = np.asarray(sampling_var, float)
    ok = np.isfinite(v) & np.isfinite(s)
    mu = float(np.mean(v[ok]))
    signal = max(0.0, float(np.var(v[ok], ddof=1)) - float(np.mean(s[ok])))
    w = signal / (signal + np.where(np.isfinite(s), s, np.inf))
    return mu + w * (v - mu)


# ──────────────────────────────────────────────────────────────────────────
# Per-batter metric on an arbitrary swing subset
# ──────────────────────────────────────────────────────────────────────────

def metric_on_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-batter mda_resid on this swing subset, with the partialling refit
    inside the subset. Returns value + its sampling variance.
    """
    bat = aggregate(df)
    y = bat['mda'].to_numpy(float)
    beta, e, _, _ = fit_ols(y, bat[PITCHMIX_ONLY].to_numpy(float))
    var = exact_sampling_var(df, bat, PITCHMIX_ONLY, beta)
    return pd.DataFrame({'mda_resid': e, 'mda_resid_var': var,
                         'n': bat['n'].values}, index=bat.index)


# ──────────────────────────────────────────────────────────────────────────
# Step 2
# ──────────────────────────────────────────────────────────────────────────

def compute_chase(season: int) -> pd.Series:
    files = sorted(glob.glob(os.path.join(DATA_ROOT, f'all_pitches_{season}',
                                          '*.parquet')))
    df = pd.concat([pd.read_parquet(f, columns=['batter', 'zone',
                                                'description'])
                    for f in files], ignore_index=True)
    df = df[~df['description'].isin(BUNT_DESC)]
    oz = df[df['zone'] > 9].copy()
    oz['swung'] = oz['description'].isin(ALL_SWINGS).astype(float)
    g = oz.groupby('batter')['swung']
    out = g.mean()
    out.index = out.index.astype(str)
    return out


def step2(variant: str, eval_df: pd.DataFrame, season: int) -> pd.DataFrame:
    swings = build_swings(variant, eval_df)
    m = metric_on_subset(swings)

    g = swings.groupby('batter')
    pub = pd.DataFrame({
        'whiff_pct': 1.0 - g['_z_contact'].mean(),
        'n_swings': g.size(),
    })
    inz = swings.merge(eval_df[['row_key', 'in_zone']], on='row_key',
                       how='left')
    iz = inz[inz['in_zone'] == True]
    pub['zone_contact_pct'] = iz.groupby('batter')['_z_contact'].mean()
    pub['chase_pct'] = compute_chase(season).reindex(pub.index)

    stats_csv = os.path.join(DATA_ROOT, f'batter_stats_{season}.csv')
    bs = pd.read_csv(stats_csv, low_memory=False)
    bs['batter'] = bs['batter_id'].astype('Int64').astype(str)
    bs = bs.drop_duplicates('batter').set_index('batter')
    for col, name in [('ev_ev95percent', 'hard_hit_pct'),
                      ('xst_est_woba', 'xwoba'),
                      ('pct_squared_up_rate', 'squared_up_pctl'),
                      ('pct_chase_percent', 'chase_pctl')]:
        if col in bs.columns:
            pub[name] = bs[col].reindex(pub.index)

    joined = pub.join(m[['mda_resid']])
    rows = []
    for c in ['whiff_pct', 'zone_contact_pct', 'chase_pct', 'hard_hit_pct',
              'xwoba', 'squared_up_pctl', 'chase_pctl']:
        if c not in joined.columns:
            continue
        d = joined[['mda_resid', c]].dropna()
        if len(d) < 20:
            continue
        pr = stats.pearsonr(d['mda_resid'], d[c])
        sr = stats.spearmanr(d['mda_resid'], d[c])
        rows.append(dict(variant=variant, metric='mda_resid', public_stat=c,
                         n=len(d), pearson_r=float(pr.statistic),
                         pearson_p=float(pr.pvalue),
                         spearman_r=float(sr.statistic),
                         is_percentile_rank=c.endswith('_pctl'),
                         abs_max=float(max(abs(pr.statistic),
                                           abs(sr.statistic)))))
    return pd.DataFrame(rows).sort_values('abs_max', ascending=False)


# ──────────────────────────────────────────────────────────────────────────
# Step 3
# ──────────────────────────────────────────────────────────────────────────

def cv_rmse(X: np.ndarray, y: np.ndarray) -> float:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.empty(len(y))
    for tr, te in kf.split(X):
        pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return rmse(pred, y)


def step3(variant: str, eval_df: pd.DataFrame, season: int) -> pd.DataFrame:
    swings = build_swings(variant, eval_df).merge(
        eval_df[['row_key'] + CHRON_KEYS], on='row_key', how='left')
    swings = swings.sort_values(CHRON_KEYS, kind='mergesort')

    # Full-season targets.
    g = swings.groupby('batter')
    targets = pd.DataFrame({'contact_pct': g['_z_contact'].mean(),
                            'n_full': g.size()})
    bs = pd.read_csv(os.path.join(DATA_ROOT, f'batter_stats_{season}.csv'),
                     low_memory=False)
    bs['batter'] = bs['batter_id'].astype('Int64').astype(str)
    bs = bs.drop_duplicates('batter').set_index('batter')
    targets['xwoba'] = bs['xst_est_woba'].reindex(targets.index)
    targets['hard_hit_pct'] = bs['ev_ev95percent'].reindex(targets.index)

    rank = swings.groupby('batter').cumcount() + 1
    n_tot = swings.groupby('batter')['batter'].transform('size')

    rows = []
    for frac in SAMPLE_FRACTIONS:
        keep = np.ceil(n_tot * frac).astype(int).clip(lower=1)
        early = swings[rank <= keep]

        m = metric_on_subset(early)
        eg = early.groupby('batter')['_z_contact']
        est = pd.DataFrame({'raw': eg.mean(), 'n_early': eg.size()})
        est = est.join(m[['mda_resid', 'mda_resid_var']])
        est['raw_shrunk'] = shrink_rate(est['raw'], est['n_early'])
        est['mda_shrunk'] = shrink_normal(est['mda_resid'],
                                          est['mda_resid_var'])

        for tname in ['contact_pct', 'xwoba', 'hard_hit_pct']:
            d = est.join(targets[[tname]]).dropna(
                subset=['raw_shrunk', 'mda_shrunk', tname])
            if len(d) < 50:
                continue
            y = d[tname].to_numpy(float)
            x_raw = d[['raw_shrunk']].to_numpy(float)
            x_both = d[['raw_shrunk', 'mda_shrunk']].to_numpy(float)

            r_direct = (rmse(d['raw_shrunk'], y) if tname == 'contact_pct'
                        else np.nan)
            r_fit = cv_rmse(x_raw, y)
            r_both = cv_rmse(x_both, y)
            rows.append(dict(
                variant=variant, target=tname, sample_fraction=frac,
                n_batters=int(len(d)),
                median_early_swings=float(d['n_early'].median()),
                rmse_raw_direct_shrunk=r_direct,
                rmse_raw_fitted=r_fit,
                rmse_raw_plus_mda=r_both,
                pct_improvement=100.0 * (r_fit - r_both) / r_fit,
                partial_r_mda=float(pd.Series(
                    y - LinearRegression().fit(x_raw, y).predict(x_raw)).corr(
                    d['mda_shrunk'].reset_index(drop=True)))))
    return pd.DataFrame(rows)


def plot_step3(df: pd.DataFrame, path: str):
    targets = list(dict.fromkeys(df['target']))
    fig, axes = plt.subplots(1, len(targets), figsize=(5.3 * len(targets), 4.3),
                             squeeze=False, constrained_layout=True)
    for i, t in enumerate(targets):
        ax = axes[0][i]
        s = df[df['target'] == t].sort_values('sample_fraction')
        ax.plot(s['sample_fraction'], s['rmse_raw_fitted'], color=BLUE, lw=2,
                marker='o', ms=4, label='early raw rate (EB-shrunk, CV-fitted)')
        ax.plot(s['sample_fraction'], s['rmse_raw_plus_mda'], color=RED, lw=2,
                marker='s', ms=4, label='+ early mda_resid')
        if s['rmse_raw_direct_shrunk'].notna().any():
            ax.plot(s['sample_fraction'], s['rmse_raw_direct_shrunk'],
                    color=GRAY, lw=1.5, ls='--', marker='^', ms=4,
                    label='EB-shrunk raw, unfitted')
        ax.set_title(f'target: full-season {t}', fontsize=11)
        ax.set_xlabel('fraction of season in the early sample')
        ax.grid(alpha=0.3, linewidth=0.5)
        if i == 0:
            ax.set_ylabel('CV RMSE')
            ax.legend(fontsize=8)
    fig.suptitle(f'Step 3 — incremental predictive value of mda_resid '
                 f'({EVAL_SEASON})', fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def main():
    ap = argparse.ArgumentParser(description='Steps 2 and 3 of the MDA follow-up')
    ap.add_argument('--variant', default=DEFAULT_VARIANT)
    ap.add_argument('--skip-step2', action='store_true')
    ap.add_argument('--skip-step3', action='store_true')
    args = ap.parse_args()

    _, eval_df = load_train_and_eval(TRAIN_SEASON, EVAL_SEASON, verbose=False)
    pd.set_option('display.width', 230)

    if not args.skip_step2:
        r2 = step2(args.variant, eval_df, EVAL_SEASON)
        p = os.path.join(OUT_DIR, 'redundancy_correlations.csv')
        r2.to_csv(p, index=False)
        print('=== Step 2: redundancy of mda_resid vs public stats ===\n')
        print(r2.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
        print(f'\n-> {p}\n')

    if not args.skip_step3:
        r3 = step3(args.variant, eval_df, EVAL_SEASON)
        p = os.path.join(OUT_DIR, 'mda_stabilization.csv')
        r3.to_csv(p, index=False)
        print('=== Step 3: incremental predictive value ===\n')
        print(r3.to_string(index=False, float_format=lambda v: f'{v:,.5f}'))
        plot_step3(r3, os.path.join(PLOT_DIR, 'mda_stabilization.png'))
        print(f'-> {p}')


if __name__ == '__main__':
    main()
