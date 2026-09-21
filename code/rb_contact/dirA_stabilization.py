"""
dirA_stabilization.py
======================
Direction A: predictive-stabilization test.

Why this exists
---------------
Step 5 compared RB-Contact% against raw contact% on *same-sample fit* and
found RB-Contact% did not win (validation_report.md: discrimination 0.8746
vs 0.8775). But that is not the claim the reference paper (Daly-Grafstein &
Bornn, RB-FG%) actually made. Their headline (their Figure 6) was a
*data-efficiency* claim: estimated from a small fraction of a season,
RB-FG% predicts the player's full-season rate with lower RMSE than the raw
rate does. The two tests can disagree -- a model-based estimator can be
identically discriminative in a full sample yet far less noisy in a small
one, because it borrows strength from the pitch/swing context of each
individual attempt instead of relying on the attempt's binary outcome.

This script runs that test, and nothing else. No refitting: it reads the
existing out-of-fold p_contact.

Designs
-------
`full_season` (the reference paper's design)
    Target = the batter's complete-season raw contact%. The early raw
    estimate is a *subset* of its own target, which mechanically helps raw
    (shared swings are shared noise). Reported because it is the exact
    design being replicated.

`holdout` (the honest design, and our stand-in for the paper's Step-6
           year-over-year robustness check)
    Target = the batter's raw contact% over the swings NOT in the early
    sample. No overlap, so neither estimator gets a free ride.

    A true year-over-year split is not runnable here: data/ contains
    pitch-level Statcast for 2025 only (all_pitches_2025/, March-September).
    `--design yoy` is implemented and will run if a second season is ever
    added; with one season it exits with an explanation rather than
    silently substituting something else.

Estimators
----------
    early_raw     mean(is_contact) over the early sample
    early_rb      mean(p_contact)  over the early sample      <- RB-Contact%
    early_raw_eb  early_raw, Beta-Binomial shrunk to the league mean
    early_rb_eb   early_rb,  Beta-Binomial shrunk to the league mean

Model variants (`--variant`), and the leak they exist to measure
----------------------------------------------------------------
`hybrid` (default) uses the pipeline's cached `p_contact`, from the hybrid
GBM. Two of that model's features -- `predicted_timing` and
`predicted_offset` -- come from MERFs carrying a **batter random intercept
fit on the batter's whole season**. So a swing from April already encodes
that batter's season-long timing baseline, and "estimated from the first 5%
of the season" is not literally true of the information content. The
stabilization advantage would be overstated by however much that baseline
is doing.

`raw` re-scores every swing with the Step 2 raw-inputs GBM (`X_raw` only:
bat-tracking kinematics, pitch characteristics, handedness -- no MERF-derived
feature, no batter identity anywhere), out-of-fold on the same folds. Nothing
in it can carry a batter-season effect, so its curve is the honest lower
bound on the data-efficiency claim. Both are reported; the `raw` curve is the
one to quote if the claim is challenged.

The shrunk pair is supplementary, not part of the spec's two-estimator
comparison, but it is the fair fight: at a 5% sample an unshrunk raw rate is
~15 swings and is dominated by pure binomial noise that *any* shrinkage
fixes. Including it separates "RB helps" from "regressing to the mean
helps", which are different claims and would otherwise be confounded.

Outputs (out/rb_contact/followups/)
-----------------------------------
    stabilization_curve.csv   RMSE by sample fraction x estimator x design
    stabilization_curve.png   Figure-6-style plot, one panel per design
    direction_a_verdict.md    written verdict
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from followup_utils import (load_swing_frame, chronological_prefix, rmse,
                            md_table, FOLLOWUP_DIR, CHRON_KEYS)
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE, RED, GREEN, ORANGE = '#2563EB', '#DC2626', '#16A34A', '#EA580C'

SAMPLE_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MIN_FULL_SEASON_SWINGS = 50
MIN_HOLDOUT_SWINGS = 25  # holdout target needs enough swings to be a target

ESTIMATORS = {
    'early_raw_contact_pct': ('raw contact%', BLUE, '-'),
    'early_RB_contact_pct': ('RB-Contact%', RED, '-'),
    'early_raw_contact_pct_eb': ('raw contact% (EB-shrunk)', BLUE, '--'),
    'early_RB_contact_pct_eb': ('RB-Contact% (EB-shrunk)', RED, '--'),
}


RAW_PROBS_NAME = 'oof_contact_probs_raw.parquet'


def ensure_raw_probs() -> pd.DataFrame:
    """
    Out-of-fold P(contact) from the Step 2 **raw-inputs** GBM -- X_raw only,
    no predicted_timing/predicted_offset, so no MERF batter random effect can
    reach the probability. Cached to out/rb_contact/oof_contact_probs_raw.parquet.

    Written here rather than calling step5's generate_oof_contact_probs()
    because that function hardcodes its output path to
    oof_contact_probs.parquet and would overwrite the pipeline's hybrid
    artifact.
    """
    from oof_timing_offset import prepare_data as _prep
    from model_utils import assemble_design, make_gbm, CONTINUOUS_RAW, BINARY_RAW

    path = os.path.join(os.path.dirname(FOLLOWUP_DIR), RAW_PROBS_NAME)
    if os.path.exists(path):
        print(f'Using cached {RAW_PROBS_NAME}')
        return pd.read_parquet(path)

    print(f'Generating {RAW_PROBS_NAME} (5 raw-feature GBM fits)...')
    df = _prep()
    oof = pd.read_parquet(os.path.join(os.path.dirname(FOLLOWUP_DIR),
                                       'oof_predictions.parquet'))[['row_key', 'fold']]
    df = df.merge(oof, on='row_key', how='inner')

    needed = CONTINUOUS_RAW + BINARY_RAW + ['pitch_type', 'is_contact', 'fold']
    sub = df.dropna(subset=needed).copy()
    pt_categories = sorted(sub['pitch_type'].unique())

    preds = []
    for fold in sorted(sub['fold'].unique()):
        train, test = sub[sub['fold'] != fold], sub[sub['fold'] == fold]
        model = make_gbm()
        model.fit(assemble_design(train, [], pt_categories),
                  train['is_contact'].astype(int))
        p = model.predict_proba(assemble_design(test, [], pt_categories))[:, 1]
        preds.append(pd.DataFrame({'row_key': test['row_key'].values, 'p_contact': p}))
        print(f'  fold {fold}: n={len(test):,}')

    out = sub[['row_key', 'batter', 'game_date', 'game_pk', 'at_bat_number',
               'pitch_number', 'is_contact']].merge(
        pd.concat(preds, ignore_index=True), on='row_key', how='left')
    out.to_parquet(path, index=False)
    print(f'-> {path}')
    return out


def shrink(values: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Empirical-Bayes Beta-Binomial shrinkage using the prior fit to *this*
    estimator at *this* sample fraction. Fitting the prior per-(estimator,
    fraction) rather than once globally is deliberate: the method-of-moments
    prior is fit to the observed spread, which is inflated by sampling noise
    at small fractions, so a prior borrowed from the full season would
    under-shrink exactly where shrinkage matters most.
    """
    try:
        alpha, beta = fit_beta_prior_moments(values, weights=n)
    except ValueError:
        # Population too homogeneous for a Beta prior at this fraction --
        # fall back to no shrinkage rather than dropping the curve point.
        return values
    return shrink_beta_binomial(values, n, alpha, beta)


def build_curve(df: pd.DataFrame, design: str, variant: str = 'hybrid') -> pd.DataFrame:
    """RMSE of each early estimator against the design's target, by fraction."""
    counts = df.groupby('batter').size()
    eligible = counts[counts >= MIN_FULL_SEASON_SWINGS].index
    df = df[df['batter'].isin(eligible)].copy()

    full_season = df.groupby('batter').agg(
        target_full=('is_contact', 'mean'), n_full=('is_contact', 'size'))

    rows = []
    for frac in SAMPLE_FRACTIONS:
        early = chronological_prefix(df, frac)
        est = early.groupby('batter').agg(
            early_raw_contact_pct=('is_contact', 'mean'),
            early_RB_contact_pct=('p_contact', 'mean'),
            n_early=('is_contact', 'size'),
        )
        est['early_raw_contact_pct_eb'] = shrink(
            est['early_raw_contact_pct'].values, est['n_early'].values)
        est['early_RB_contact_pct_eb'] = shrink(
            est['early_RB_contact_pct'].values, est['n_early'].values)

        if design == 'full_season':
            tgt = full_season['target_full']
            n_target = full_season['n_full']
        else:  # holdout: complement of the early sample
            early_keys = set(early['row_key'])
            late = df[~df['row_key'].isin(early_keys)]
            late_g = late.groupby('batter')['is_contact']
            tgt = late_g.mean()
            n_target = late_g.size()
            keep = n_target[n_target >= MIN_HOLDOUT_SWINGS].index
            tgt, n_target = tgt.loc[keep], n_target.loc[keep]

        m = est.join(tgt.rename('target'), how='inner').dropna(subset=['target'])
        m = m.join(n_target.rename('n_target'), how='left')

        for col in ESTIMATORS:
            rows.append(dict(
                variant=variant, design=design, sample_fraction=frac, estimator=col,
                rmse=rmse(m[col], m['target']),
                mae=float(np.nanmean(np.abs(m[col] - m['target']))),
                pearson_r=float(m[col].corr(m['target'])),
                n_batters=len(m),
                mean_early_swings=float(m['n_early'].mean()),
                median_early_swings=float(m['n_early'].median()),
                mean_target_swings=float(m['n_target'].mean()),
            ))
        print(f'  [{variant}/{design}] frac={frac:.2f}  n_batters={len(m):,}  '
              f'median early swings={m["n_early"].median():.0f}  '
              f'raw RMSE={rows[-4]["rmse"]:.5f}  RB RMSE={rows[-3]["rmse"]:.5f}')
    return pd.DataFrame(rows)


def run_yoy(df: pd.DataFrame, variant: str = 'hybrid') -> pd.DataFrame | None:
    """
    Spec step 6: first half of season N predicting full season N+1. Requires
    two seasons of pitch-level data; returns None (with an explanation) if
    only one season is present.
    """
    seasons = sorted(df['game_date'].dt.year.unique())
    if len(seasons) < 2:
        print(f'\n[yoy] Only season(s) {seasons} present in data/all_pitches_*/ -- '
              f'a year-over-year split needs two. Skipping; the `holdout` design '
              f'is the disjoint-sample robustness check that IS runnable here.')
        return None

    rows = []
    for s_n, s_next in zip(seasons[:-1], seasons[1:]):
        cur = df[df['game_date'].dt.year == s_n]
        nxt = df[df['game_date'].dt.year == s_next]
        target = nxt.groupby('batter')['is_contact'].agg(['mean', 'size'])
        target = target[target['size'] >= MIN_FULL_SEASON_SWINGS]
        for frac in SAMPLE_FRACTIONS:
            early = chronological_prefix(cur, frac)
            est = early.groupby('batter').agg(
                early_raw_contact_pct=('is_contact', 'mean'),
                early_RB_contact_pct=('p_contact', 'mean'),
                n_early=('is_contact', 'size'))
            est['early_raw_contact_pct_eb'] = shrink(
                est['early_raw_contact_pct'].values, est['n_early'].values)
            est['early_RB_contact_pct_eb'] = shrink(
                est['early_RB_contact_pct'].values, est['n_early'].values)
            m = est.join(target['mean'].rename('target'), how='inner')
            for col in ESTIMATORS:
                rows.append(dict(
                    variant=variant, design=f'yoy_{s_n}_to_{s_next}', sample_fraction=frac,
                    estimator=col, rmse=rmse(m[col], m['target']),
                    mae=float(np.nanmean(np.abs(m[col] - m['target']))),
                    pearson_r=float(m[col].corr(m['target'])),
                    n_batters=len(m), mean_early_swings=float(m['n_early'].mean()),
                    median_early_swings=float(m['n_early'].median()),
                    mean_target_swings=float(target['size'].mean())))
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, path: str):
    designs = list(dict.fromkeys(curves['design']))
    variants = list(dict.fromkeys(curves['variant']))
    fig, axes = plt.subplots(len(variants), len(designs),
                             figsize=(6.2 * len(designs), 5.0 * len(variants)),
                             squeeze=False)
    titles = {
        'full_season': ("Target: batter's FULL-season raw contact%\n"
                        "(reference paper's design; early raw sample is a\n"
                        "subset of its own target)"),
        'holdout': ("Target: batter's raw contact% on the REMAINING swings\n"
                    "(disjoint samples -- no shared-noise advantage for raw)"),
    }
    vtitle = {'hybrid': 'hybrid GBM p_contact -- carries a full-season batter random effect '
                        'via predicted_timing/predicted_offset (upper bound)',
              'raw': 'raw-inputs GBM p_contact -- X_raw only, no batter random effect '
                     'can reach the probability (quote this one)'}
    for vi, variant in enumerate(variants):
        for di, design in enumerate(designs):
            ax = axes[vi][di]
            sub = curves[(curves['design'] == design) & (curves['variant'] == variant)]
            for col, (label, color, ls) in ESTIMATORS.items():
                s = sub[sub['estimator'] == col].sort_values('sample_fraction')
                ax.plot(s['sample_fraction'], s['rmse'], ls, marker='o', ms=5,
                        color=color, label=label, alpha=0.95 if ls == '-' else 0.6)
            ax.set_xlabel('Fraction of season used for the early estimate')
            ax.set_ylabel('RMSE vs. target contact%')
            ax.set_title(f'[{variant}] ' + titles.get(design, design),
                         fontsize=9.5, fontweight='bold')
            ax.legend(fontsize=8.5)
            ax.grid(alpha=0.3)
    # Variant explanations go in a footnote rather than as rotated row
    # labels: tight_layout repositions the axes after they'd be placed, so
    # anything anchored to an axes position lands on top of the axis labels.
    footnote = '\n'.join(f'[{v}]  {vtitle.get(v, v)}' for v in variants)
    plt.tight_layout(rect=[0, 0.055 * len(variants), 1, 0.955])
    fig.suptitle('Direction A -- predictive stabilization of RB-Contact% vs. raw contact%',
                 fontweight='bold', y=0.985, fontsize=14)
    fig.text(0.5, 0.005, footnote, ha='center', va='bottom', fontsize=9.5,
             style='italic', color='#444444')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'-> {path}')


def write_verdict(curves: pd.DataFrame, path: str):
    lines = ['# Direction A -- Predictive-Stabilization Test: Verdict\n']
    lines.append(
        'Question: estimated from the first *f* of a batter\'s season (chronologically), '
        'does RB-Contact% predict that batter\'s contact rate better than the raw '
        'contact rate does? This is the reference paper\'s actual headline claim '
        '(their Figure 6), as distinct from the same-sample fit comparison in '
        '`validation_report.md`.\n')
    lines.append(f'Population: batters with >= {MIN_FULL_SEASON_SWINGS} swings in the '
                 f'2025 season, all out-of-fold per-swing probabilities.\n')
    lines.append(
        '**Two model variants are reported, and the difference between them matters.** '
        'The `hybrid` model is the pipeline\'s existing contact model, whose '
        '`predicted_timing` / `predicted_offset` features come from MERFs carrying a '
        'batter random intercept fit on the batter\'s *whole season*. An April swing '
        'therefore already encodes that batter\'s season-long baseline, so a `hybrid` '
        'advantage at f=0.05 is not purely a data-efficiency result. The `raw` model '
        'uses `X_raw` only -- bat-tracking kinematics, pitch characteristics, '
        'handedness, no MERF feature and no batter identity -- so nothing in it can '
        'carry a batter-season effect. **Quote the `raw` curve**; the `hybrid` curve '
        'is an upper bound.\n')

    for variant in dict.fromkeys(curves['variant']):
      for design in dict.fromkeys(curves['design']):
        sub = curves[(curves['design'] == design) & (curves['variant'] == variant)]
        if sub.empty:
            continue
        lines.append(f'\n## Model `{variant}` / design `{design}`\n')
        piv = sub.pivot(index='sample_fraction', columns='estimator', values='rmse')
        piv = piv[[c for c in ESTIMATORS if c in piv.columns]]
        piv.insert(0, 'median_early_swings',
                   sub.groupby('sample_fraction')['median_early_swings'].first())
        piv['RB_minus_raw'] = piv['early_RB_contact_pct'] - piv['early_raw_contact_pct']
        piv['RB_minus_raw_pct'] = 100 * piv['RB_minus_raw'] / piv['early_raw_contact_pct']
        lines.append('RMSE by sample fraction (lower is better; '
                     '`RB_minus_raw` < 0 means RB-Contact% wins):\n')
        lines.append(md_table(piv.reset_index().round(5), floatfmt=5))

        d = piv['RB_minus_raw']
        best_frac = d.idxmin()
        wins = (d < 0).sum()
        lines.append(f'\n- RB-Contact% has lower RMSE at **{wins} of {len(d)}** sample '
                     f'fractions.')
        lines.append(f'- Largest RB advantage: **{d.min():+.5f}** RMSE '
                     f'({piv.loc[best_frac, "RB_minus_raw_pct"]:+.1f}%) at '
                     f'f={best_frac:.2f} (median {piv.loc[best_frac, "median_early_swings"]:.0f} '
                     f'early swings).')
        lines.append(f'- At the smallest fraction (f=0.05): raw RMSE '
                     f'{piv.loc[0.05, "early_raw_contact_pct"]:.5f} vs RB '
                     f'{piv.loc[0.05, "early_RB_contact_pct"]:.5f} '
                     f'({piv.loc[0.05, "RB_minus_raw_pct"]:+.1f}%).')
        lines.append(f'- At the largest fraction (f=0.50): raw RMSE '
                     f'{piv.loc[0.50, "early_raw_contact_pct"]:.5f} vs RB '
                     f'{piv.loc[0.50, "early_RB_contact_pct"]:.5f} '
                     f'({piv.loc[0.50, "RB_minus_raw_pct"]:+.1f}%).')

        eb_d = piv['early_RB_contact_pct_eb'] - piv['early_raw_contact_pct_eb']
        lines.append(f'- With both estimators Beta-Binomial shrunk (supplementary, '
                     f'controls for "shrinkage alone helps"): RB wins at '
                     f'{(eb_d < 0).sum()} of {len(eb_d)} fractions, best '
                     f'{eb_d.min():+.5f} RMSE.')
        # Which of the four estimators is actually best at each fraction? The
        # spec's two-way comparison can't distinguish "the model helps" from
        # "any shrinkage helps"; this four-way ranking can.
        four = piv[list(ESTIMATORS)]
        best = four.idxmin(axis=1)
        lines.append(f'- Best of all four estimators by fraction: ' +
                     ', '.join(f'f={f:.2f} -> `{b}`' for f, b in best.items()) + '.')
        # Convergence point: first fraction at/after which |delta| is < 5% of raw RMSE
        rel = (d.abs() / piv['early_raw_contact_pct'])
        conv = rel[rel < 0.05]
        if len(conv) and len(conv) < len(rel):
            lines.append(f'- Curves converge (|delta| < 5% of raw RMSE) from '
                         f'f={conv.index[0]:.2f} onward.')
        elif len(conv) == len(rel):
            lines.append('- Curves are within 5% of each other at *every* '
                         'fraction tested -- no separation anywhere.')

    lines.append('\n## Year-over-year robustness check (spec step 6)\n')
    lines.append('Not runnable: `data/` holds pitch-level Statcast for the 2025 season '
                 'only (`all_pitches_2025/`, March-September). A first-half-of-N -> '
                 'full-season-N+1 split needs two seasons. `dirA_stabilization.py '
                 '--design yoy` implements it and will produce the curve as soon as a '
                 'second season is downloaded. The `holdout` design above is the '
                 'disjoint-sample check that is runnable on one season: it removes the '
                 'shared-noise advantage the `full_season` target hands the raw '
                 'estimator, which is the main confound a YoY split would also remove.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--design', choices=['both', 'full_season', 'holdout', 'yoy'],
                    default='both')
    ap.add_argument('--variant', choices=['both', 'hybrid', 'raw'], default='both',
                    help='which contact model supplies p_contact; see module docstring')
    args = ap.parse_args()

    cols = ['row_key', 'batter', 'game_date', 'game_pk', 'at_bat_number',
            'pitch_number', 'is_contact', 'p_contact']

    sources = {}
    if args.variant in ('both', 'hybrid'):
        hyb = load_swing_frame(with_counts=False)
        sources['hybrid'] = hyb.dropna(subset=['p_contact'])[cols].copy()
    if args.variant in ('both', 'raw'):
        raw = ensure_raw_probs()
        raw['game_date'] = pd.to_datetime(raw['game_date'])
        sources['raw'] = raw.dropna(subset=['p_contact'])[cols].copy()

    designs = ['full_season', 'holdout'] if args.design == 'both' else [args.design]
    frames = []
    for variant, df in sources.items():
        print(f'\nSwings with an out-of-fold p_contact [{variant}]: {len(df):,} '
              f'({df["batter"].nunique():,} batters)')
        for design in designs:
            if design == 'yoy':
                yoy = run_yoy(df, variant)
                if yoy is not None:
                    frames.append(yoy)
                continue
            print(f'\n=== {variant} / {design} ===')
            frames.append(build_curve(df, design, variant))

        if args.design == 'both':
            yoy = run_yoy(df, variant)
            if yoy is not None:
                frames.append(yoy)

    curves = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(FOLLOWUP_DIR, 'stabilization_curve.csv')
    curves.to_csv(csv_path, index=False)
    print(f'\n-> {csv_path}')

    plot_curves(curves, os.path.join(FOLLOWUP_DIR, 'stabilization_curve.png'))
    write_verdict(curves, os.path.join(FOLLOWUP_DIR, 'direction_a_verdict.md'))


if __name__ == '__main__':
    main()
