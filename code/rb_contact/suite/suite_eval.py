"""
suite_eval.py
==============
Directions A / B / C from the rb_contact framework, re-run against the
2026 evaluation season for each model variant.

Why 2026 changes what these analyses mean
------------------------------------------
In the original pipeline every quantity was out-of-fold *within* 2025, so
"held out" meant held out from a fold. Here the stage-1 and stage-2 models are
fit on 2025 only and 2026 is scored by them, so no 2026 row took part in any
fit. The batter random intercepts are the one channel by which 2025 information
reaches a 2026 prediction, which is exactly why every analysis is run twice:

    carry    2025 random intercept applied to the batter's 2026 swings
    context  random intercept suppressed; pitch/swing context only

For the RF variant the two are identical by construction, which makes it the
reference line for how much of any advantage is batter identity rather than
pitch-context modelling.

Direction A -- stabilization / data efficiency
    Does RB-Contact% estimated from a small early sample predict the
    full-season rate better than the raw rate does? Three designs:
      full_season  early 2026 -> full 2026        (the reference paper's, and
                                                   optimistic: the early sample
                                                   is inside its own target)
      holdout      early 2026 -> rest of 2026     (disjoint, honest)
      yoy          early 2025 -> full 2026        (now runnable for the first
                                                   time, with two seasons)

Direction B -- Miss Distance Added
    MDA_batter = mean(actual miss_distance - predicted miss distance).
    Negative is good: closer to the ball than the pitch context predicted.
    Scored by the Franks et al. discrimination and two-period stability
    metrics, against raw contact% and whiff% as reference stats.

    dirB's original finding was that measuring this against a prediction that
    already contains the batter's own random intercept yields a null statistic,
    because the intercept pins each batter's residual sum near zero. The
    `context` columns are the fix and are the ones to read; `carry` is retained
    to show the failure mode still reproduces.

Direction C -- per-swing diagnostics
    A queryable per-swing residual table for 2026, plus pattern tables
    breaking each batter's timing and offset residuals down by pitch type,
    count and zone.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suite_config import (VARIANT_ORDER, TRAIN_SEASON, EVAL_SEASON, OUT_DIR,
                          PLOT_DIR, DATA_ROOT, eval_scored_path, oof_path,
                          probs_path)
from suite_stage2 import RAW_TAG
from followup_utils import (walk_counts, discrimination,
                            discrimination_bootstrap, stability_two_period,
                            rate_stat, mean_stat, chronological_prefix, rmse,
                            md_table, CHRON_KEYS)
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY = '#2563EB', '#DC2626', '#16A34A', '#6B7280'

SAMPLE_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MIN_FULL_SEASON_SWINGS = 50
MIN_HOLDOUT_SWINGS = 25
MIN_SWINGS_METRIC = 100
MODES = ['carry', 'context']


# ──────────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────────

def reconstruct_counts_season(season: int) -> pd.DataFrame:
    """
    Ball/strike count entering each pitch, for one season.

    Season-parameterised twin of followup_utils.reconstruct_counts (which is
    hardcoded to the 2025 directory). Reuses walk_counts unchanged -- that is
    the part with the logic worth not duplicating. Reads the raw parquets
    rather than the swing frame because the called strikes and balls the swing
    filter drops are exactly what moves the count.
    """
    cols = ['game_pk', 'at_bat_number', 'pitch_number', 'description']
    files = sorted(glob.glob(os.path.join(DATA_ROOT, f'all_pitches_{season}',
                                          '*.parquet')))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'],
                        kind='mergesort')
    ab_id = (df['game_pk'].astype(np.int64) * 1000 +
             df['at_bat_number'].astype(np.int64)).values
    balls, strikes = walk_counts(df['description'].values, ab_id)
    df['balls'], df['strikes'] = balls, strikes
    df['count_str'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    df['row_key'] = (df['game_pk'].astype(str) + '_' +
                     df['at_bat_number'].astype(str) + '_' +
                     df['pitch_number'].astype(str))
    return df[['row_key', 'balls', 'strikes', 'count_str']]


def load_variant_eval(variant: str, counts: pd.DataFrame) -> pd.DataFrame:
    """2026 swing frame for one variant, with predictions, probs and residuals."""
    df = pd.read_parquet(eval_scored_path(variant))
    probs = probs_path(variant, EVAL_SEASON)
    if os.path.exists(probs):
        p = pd.read_parquet(probs)[['row_key', 'p_contact_carry',
                                    'p_contact_context']]
        df = df.merge(p, on='row_key', how='left')
    df = df.merge(counts, on='row_key', how='left')

    for mode in MODES:
        df[f'timing_residual_{mode}'] = (df['int_y'] -
                                         df[f'predicted_timing_{mode}'])
        df[f'offset_residual_{mode}'] = (df['miss_distance_t'] -
                                         df[f'predicted_offset_{mode}'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['variant'] = variant
    return df.sort_values(CHRON_KEYS, kind='mergesort').reset_index(drop=True)


def load_train_probs(variant: str) -> pd.DataFrame | None:
    path = probs_path(variant, TRAIN_SEASON)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df.sort_values(CHRON_KEYS, kind='mergesort').reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Direction A -- stabilization
# ──────────────────────────────────────────────────────────────────────────

def _shrink(values, n):
    try:
        alpha, beta = fit_beta_prior_moments(np.asarray(values, float),
                                             weights=np.asarray(n, float))
    except ValueError:
        return np.asarray(values, float)
    return shrink_beta_binomial(np.asarray(values, float),
                                np.asarray(n, float), alpha, beta)


def _estimator_rows(early, target, label, variant, mode, design, frac):
    est = early.groupby('batter').agg(
        early_raw=('is_contact', 'mean'),
        early_rb=(label, 'mean'),
        n_early=('is_contact', 'size'))
    est['early_raw_eb'] = _shrink(est['early_raw'], est['n_early'])
    est['early_rb_eb'] = _shrink(est['early_rb'], est['n_early'])
    m = est.join(target.rename('target'), how='inner')
    rows = []
    for col, pretty in [('early_raw', 'raw contact%'),
                        ('early_rb', 'RB-Contact%'),
                        ('early_raw_eb', 'raw contact% (EB)'),
                        ('early_rb_eb', 'RB-Contact% (EB)')]:
        rows.append(dict(variant=variant, mode=mode, design=design,
                         sample_fraction=frac, estimator=pretty,
                         rmse=rmse(m[col], m['target']),
                         mae=float(np.nanmean(np.abs(m[col] - m['target']))),
                         pearson_r=float(m[col].corr(m['target'])),
                         n_batters=int(len(m)),
                         median_early_swings=float(m['n_early'].median())))
    return rows


def direction_a(ev: pd.DataFrame, tr: pd.DataFrame | None,
                variant: str) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        pcol = f'p_contact_{mode}'
        if pcol not in ev.columns or ev[pcol].notna().sum() == 0:
            continue
        sub = ev.dropna(subset=[pcol, 'is_contact']).copy()

        full = sub.groupby('batter')['is_contact'].agg(['mean', 'size'])
        full = full[full['size'] >= MIN_FULL_SEASON_SWINGS]

        for frac in SAMPLE_FRACTIONS:
            early = chronological_prefix(sub, frac)
            rows += _estimator_rows(early, full['mean'], pcol, variant, mode,
                                    'full_season', frac)

            # holdout: target is the swings NOT in the early sample
            early_keys = set(early['row_key'])
            rest = sub[~sub['row_key'].isin(early_keys)]
            tgt = rest.groupby('batter')['is_contact'].agg(['mean', 'size'])
            tgt = tgt[tgt['size'] >= MIN_HOLDOUT_SWINGS]
            rows += _estimator_rows(early, tgt['mean'], pcol, variant, mode,
                                    'holdout', frac)

    # yoy: early 2025 -> full 2026. Computed once, outside the mode loop: the
    # source estimate is the 2025 out-of-fold probability, which has no
    # carry/context distinction (within-season OOF always carries the batter's
    # own random effect). Labelled 'context' so it lands on the plotted slice.
    if tr is not None and 'p_contact' in tr.columns:
        base = ev.dropna(subset=['is_contact'])
        full = base.groupby('batter')['is_contact'].agg(['mean', 'size'])
        full = full[full['size'] >= MIN_FULL_SEASON_SWINGS]
        src = tr.dropna(subset=['p_contact']).copy()
        for frac in SAMPLE_FRACTIONS:
            early = chronological_prefix(src, frac)
            rows += _estimator_rows(early, full['mean'], 'p_contact',
                                    variant, 'context',
                                    f'yoy_{TRAIN_SEASON}_to_{EVAL_SEASON}',
                                    frac)
    return pd.DataFrame(rows)


def plot_direction_a(curves: pd.DataFrame, path: str):
    designs = list(dict.fromkeys(curves['design']))
    variants = list(dict.fromkeys(curves['variant']))
    fig, axes = plt.subplots(len(variants), len(designs),
                             figsize=(5.4 * len(designs), 4.2 * len(variants)),
                             squeeze=False, constrained_layout=True)
    style = {'raw contact%': (BLUE, '-'), 'RB-Contact%': (RED, '-'),
             'raw contact% (EB)': (BLUE, '--'), 'RB-Contact% (EB)': (RED, '--')}
    for ri, v in enumerate(variants):
        for ci, design in enumerate(designs):
            ax = axes[ri][ci]
            sub = curves[(curves['variant'] == v) & (curves['design'] == design)
                         & (curves['mode'] == 'context')]
            if sub.empty:
                ax.set_visible(False)
                continue
            for est, g in sub.groupby('estimator'):
                c, ls = style.get(est, (GRAY, '-'))
                g = g.sort_values('sample_fraction')
                ax.plot(g['sample_fraction'], g['rmse'], color=c, ls=ls,
                        lw=2, marker='o', ms=4, label=est)
            ax.set_title(f'{v} | {design}', fontsize=10)
            ax.grid(alpha=0.3, linewidth=0.5)
            if ri == len(variants) - 1:
                ax.set_xlabel('fraction of season used for the estimate')
            if ci == 0:
                ax.set_ylabel('RMSE vs target contact%')
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f'Direction A — stabilization on {EVAL_SEASON} '
                 f'(random effects suppressed)', fontsize=13)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


# ──────────────────────────────────────────────────────────────────────────
# Direction B -- Miss Distance Added
# ──────────────────────────────────────────────────────────────────────────

def direction_b(ev: pd.DataFrame, variant: str,
                min_swings: int = MIN_SWINGS_METRIC) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = (ev.groupby('batter').size()
              .loc[lambda s: s >= min_swings].index)
    sub = ev[ev['batter'].isin(cohort)].copy()

    disc_rows, stab_rows = [], []

    def _add(name, table, swing_df=None, value_col=None):
        d = discrimination(table['value'].values, table['sampling_var'].values)
        boot = (discrimination_bootstrap(swing_df, value_col)
                if swing_df is not None and value_col else np.nan)
        disc_rows.append(dict(variant=variant, metric=name,
                              discrimination=d, discrimination_boot=boot,
                              n_batters=int(len(table)),
                              mean_value=float(table['value'].mean())))

    _add('raw_contact_pct', rate_stat(sub, 'is_contact'))
    sub['is_whiff'] = ~sub['is_contact']
    _add('raw_whiff_pct', rate_stat(sub, 'is_whiff'))

    for mode in MODES:
        col = f'offset_residual_{mode}'
        t = mean_stat(sub, col)
        _add(f'MDA_{mode}', t, sub, col)
        tcol = f'timing_residual_{mode}'
        _add(f'timing_resid_{mode}', mean_stat(sub, tcol), sub, tcol)

    # Two-period stability: first vs second half of each batter's own season.
    sub = sub.sort_values(CHRON_KEYS, kind='mergesort')
    rank = sub.groupby('batter').cumcount()
    n = sub.groupby('batter')['batter'].transform('size')
    first, second = sub[rank < n / 2], sub[rank >= n / 2]

    for name, col in ([('raw_contact_pct', 'is_contact')] +
                      [(f'MDA_{m}', f'offset_residual_{m}') for m in MODES]):
        if col == 'is_contact':
            a, b = rate_stat(first, col), rate_stat(second, col)
        else:
            a, b = mean_stat(first, col), mean_stat(second, col)
        j = a.join(b, lsuffix='_1', rsuffix='_2', how='inner').dropna()
        if len(j) < 20:
            continue
        s = stability_two_period(j['value_1'], j['value_2'],
                                 j['sampling_var_1'], j['sampling_var_2'])
        stab_rows.append(dict(variant=variant, metric=name,
                              half_to_half_r=float(j['value_1'].corr(j['value_2'])),
                              **s))

    return pd.DataFrame(disc_rows), pd.DataFrame(stab_rows)


def plot_direction_b(disc: pd.DataFrame, path: str):
    piv = disc.pivot_table(index='metric', columns='variant',
                           values='discrimination')
    # A metric with no batters above the cohort floor yields NaN discrimination
    # (happens on small subsets). Drop those rather than handing matplotlib an
    # all-NaN frame, and skip the figure entirely if nothing is left.
    piv = piv.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    piv = piv.dropna(axis=1, how='all')
    if piv.empty:
        print(f'SKIP {os.path.basename(path)}: no non-null discrimination '
              f'values (cohort floor of {MIN_SWINGS_METRIC} swings met by no '
              f'batter?)')
        return
    ax = piv.plot(kind='barh', figsize=(9, 0.55 * len(piv) + 2.5),
                  color=[BLUE, RED, GREEN][:piv.shape[1]], width=0.78)
    ax.set_xlabel('Franks discrimination  (share of between-batter variance '
                  'that is signal)')
    ax.set_ylabel('')
    ax.set_title(f'Direction B — discrimination on {EVAL_SEASON}', fontsize=12)
    ax.grid(alpha=0.3, axis='x', linewidth=0.5)
    ax.legend(title='variant', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'-> {path}')


# ──────────────────────────────────────────────────────────────────────────
# Direction C -- per-swing diagnostics
# ──────────────────────────────────────────────────────────────────────────

def direction_c(ev: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = ['row_key', 'batter', 'pitcher', 'game_date', 'pitch_type',
            'description', 'is_contact', 'in_zone', 'balls', 'strikes',
            'count_str', 'int_y', 'miss_distance_t'] + \
           [f'{q}_{m}' for q in ('predicted_timing', 'predicted_offset',
                                 'timing_residual', 'offset_residual')
            for m in MODES]
    table = ev[[c for c in keep if c in ev.columns]].copy()
    table['variant'] = variant

    rows = []
    for bucket in ['pitch_type', 'count_str', 'in_zone']:
        for resid in ['timing_residual_context', 'offset_residual_context']:
            g = table.dropna(subset=[resid]).groupby(bucket)[resid]
            agg = pd.DataFrame({'mean_residual': g.mean(), 'sd': g.std(),
                                'n': g.size()}).reset_index()
            agg = agg.rename(columns={bucket: 'bucket_value'})
            agg['bucket'] = bucket
            agg['residual'] = resid
            agg['variant'] = variant
            rows.append(agg)
    patterns = pd.concat(rows, ignore_index=True)
    patterns = patterns[patterns['n'] >= 100]
    return table, patterns


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Directions A/B/C on the eval season')
    ap.add_argument('--variants', nargs='*', default=VARIANT_ORDER)
    ap.add_argument('--skip-a', action='store_true')
    ap.add_argument('--skip-c', action='store_true')
    ap.add_argument('--min-swings', type=int, default=MIN_SWINGS_METRIC,
                    help='minimum 2026 swings for a batter to enter the '
                         'Direction B cohort')
    args = ap.parse_args()

    print(f'Reconstructing {EVAL_SEASON} counts...')
    counts = reconstruct_counts_season(EVAL_SEASON)

    a_all, b_disc, b_stab, c_pat = [], [], [], []
    for v in args.variants:
        if not os.path.exists(eval_scored_path(v)):
            print(f'SKIP {v}: eval scoring missing')
            continue
        print(f'\n{"=" * 70}\n{v}\n{"=" * 70}', flush=True)
        ev = load_variant_eval(v, counts)
        tr = load_train_probs(v)

        if not args.skip_a:
            a = direction_a(ev, tr, v)
            if not a.empty:
                a_all.append(a)
                print(f'  dirA: {len(a):,} curve points')

        d, s = direction_b(ev, v, args.min_swings)
        b_disc.append(d)
        b_stab.append(s)
        print(f'  dirB: {len(d)} discrimination rows, {len(s)} stability rows')

        if not args.skip_c:
            table, pat = direction_c(ev, v)
            table.to_parquet(
                os.path.join(OUT_DIR, f'dirC_swing_residuals_{v}.parquet'),
                index=False)
            c_pat.append(pat)
            print(f'  dirC: {len(table):,} swings, {len(pat):,} pattern rows')

    if a_all:
        curves = pd.concat(a_all, ignore_index=True)
        curves.to_csv(os.path.join(OUT_DIR, 'dirA_stabilization.csv'),
                      index=False)
        plot_direction_a(curves, os.path.join(PLOT_DIR, 'dirA_stabilization.png'))

    if b_disc:
        disc = pd.concat(b_disc, ignore_index=True)
        stab = pd.concat(b_stab, ignore_index=True)
        disc.to_csv(os.path.join(OUT_DIR, 'dirB_discrimination.csv'), index=False)
        stab.to_csv(os.path.join(OUT_DIR, 'dirB_stability.csv'), index=False)
        plot_direction_b(disc, os.path.join(PLOT_DIR, 'dirB_discrimination.png'))
        print('\n--- Direction B discrimination ---')
        print(disc.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
        print('\n--- Direction B stability ---')
        print(stab.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))

    if c_pat:
        pd.concat(c_pat, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, 'dirC_patterns.csv'), index=False)

    print('\nDone.')


if __name__ == '__main__':
    main()
