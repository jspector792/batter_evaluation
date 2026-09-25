"""
run_pitch_grading.py
=====================
Direction 2: league-wide pitch grading for contact difficulty.

Three pieces
------------
1. Region profiling. Skill (1 - BCE / BCE at the bin's own base rate) and
   mean residual, binned finely along the axes the prior round flagged as
   weak: pfx_z, plate_z, release_speed, attack_direction.

2. The sparsity control (the gate for this direction). Bin counts are plotted
   alongside skill, and skill is correlated against log bin size. Equal-count
   bins are used for the main pass so sample size is constant by
   construction; a second equal-WIDTH pass is run where n does vary, and the
   two are compared. If skill degradation tracks n, the "weak region" finding
   is a sparsity artifact. If skill degrades identically under equal-count
   bins, it is not.

3. A per-pitch difficulty score. A contact model fit on PITCH characteristics
   only -- no bat tracking, no batter identity, nothing about the swing
   taken -- so the grade describes the pitch rather than who swung at it.

       difficulty = 1 - P(contact | swing, pitch characteristics)
                  = P(whiff | swing, pitch characteristics)

   Note the conditioning: the population is swings, so this grades "how hard
   is this pitch to hit once a hitter offers at it", not swing-and-take
   decisions together. Aggregating to the pitcher therefore averages over
   swings against them, not over all their pitches -- applying a
   swing-conditioned model to taken pitches would be extrapolating to a
   population it was never fit on.

The stage-2 probability model used here
----------------------------------------
p_contact is refit for THIS project's chosen configuration -- timing from the
MERF (m1) and miss distance from the whiff-only RF (m4). The cached stage-2
probabilities from the rb_contact suite each used a single variant's own two
features, so none of them matches this pairing.

Outputs (out/exp2_pitch_grading/)
----------------------------------
  region_profile.csv        skill / BCE / mean residual / n per bin per axis
  sparsity_check.csv        skill vs bin size correlation per axis
  pitch_difficulty.csv      score distribution summary + model info
  pitcher_grades.csv        pitcher-level aggregates
  plots/region_skill.png    skill with bin-size overlay
  plots/difficulty_dist.png score distribution and its drivers
"""

import os
import sys
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE_DIR, 'code', 'exp_common'))
sys.path.insert(0, os.path.join(BASE_DIR, 'code', 'rb_contact'))
from base_table import load as load_base  # noqa: E402
from model_utils import CONTINUOUS_RAW, BINARY_RAW  # noqa: E402

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY = '#2563EB', '#DC2626', '#16A34A', '#6B7280'

OUT_DIR = os.path.join(BASE_DIR, 'out', 'exp2_pitch_grading')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_SEASON, EVAL_SEASON = 2025, 2026
EPS = 1e-6
SEED = 42
N_BINS = 20
MIN_BIN = 300

AXES = ['pfx_z', 'plate_z', 'release_speed', 'attack_direction']

# Pitch-only features: trajectory and location. No bat tracking, no batter.
PITCH_FEATURES = ['release_speed', 'pfx_x_bat_flip', 'pfx_z',
                  'plate_x_bat_flip', 'plate_z', 'same_hand']
# Full feature set for the stage-2 style model that uses the chosen stage-1
# predictions, for the region-profiling half of this direction.
HYBRID_EXTRA = ['predicted_timing', 'predicted_offset']


def bce(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(y, p):
    y = np.asarray(y, float)
    base = np.clip(y.mean(), EPS, 1 - EPS)
    b = bce(y, np.full_like(y, base))
    return float(1 - bce(y, p) / b) if b > 0 else np.nan


def design(df, cols, pt_cats):
    X = df[cols].astype(float).copy()
    pt = pd.Categorical(df['pitch_type'], categories=pt_cats)
    return pd.concat([X.reset_index(drop=True),
                      pd.get_dummies(pt, prefix='pt').reset_index(drop=True)],
                     axis=1)


def fit_calibrated(train, cols, pt_cats):
    from sklearn.model_selection import train_test_split
    fit_part, cal_part = train_test_split(train, test_size=0.15,
                                          random_state=SEED,
                                          stratify=train['is_contact'])
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                      reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                      n_jobs=-1, eval_metric='logloss', verbosity=0)
    m.fit(design(fit_part, cols, pt_cats),
          fit_part['is_contact'].astype(int))
    cal = CalibratedClassifierCV(FrozenEstimator(m), method='isotonic')
    cal.fit(design(cal_part, cols, pt_cats),
            cal_part['is_contact'].astype(int))
    return cal


def score_models(tr, ev):
    """p_contact from (a) the hybrid chosen-config model, (b) pitch-only."""
    pt_cats = sorted(set(tr['pitch_type']) | set(ev['pitch_type']))

    hyb_cols = CONTINUOUS_RAW + BINARY_RAW + HYBRID_EXTRA
    tr_h = tr.dropna(subset=hyb_cols + ['is_contact']).copy()
    ev_h = ev.dropna(subset=hyb_cols + ['is_contact']).copy()
    cal_h = fit_calibrated(tr_h, hyb_cols, pt_cats)
    ev_h['p_contact'] = cal_h.predict_proba(design(ev_h, hyb_cols, pt_cats))[:, 1]

    tr_p = tr.dropna(subset=PITCH_FEATURES + ['is_contact']).copy()
    ev_p = ev.dropna(subset=PITCH_FEATURES + ['is_contact']).copy()
    cal_p = fit_calibrated(tr_p, PITCH_FEATURES, pt_cats)
    ev_p['p_contact_pitch'] = cal_p.predict_proba(
        design(ev_p, PITCH_FEATURES, pt_cats))[:, 1]

    out = ev_h.merge(ev_p[['row_key', 'p_contact_pitch']], on='row_key',
                     how='inner')
    print(f'  hybrid model: BCE {bce(out["is_contact"], out["p_contact"]):.5f}'
          f'  skill {skill(out["is_contact"], out["p_contact"]):.4f}')
    print(f'  pitch-only  : BCE '
          f'{bce(out["is_contact"], out["p_contact_pitch"]):.5f}'
          f'  skill {skill(out["is_contact"], out["p_contact_pitch"]):.4f}')
    return out


def profile_axis(df, axis, mode='quantile', n_bins=N_BINS):
    v = df[axis].to_numpy(float)
    ok = np.isfinite(v)
    d = df[ok].copy()
    v = v[ok]
    if mode == 'quantile':
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(np.nanpercentile(v, 0.5),
                            np.nanpercentile(v, 99.5), n_bins + 1)
    idx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < MIN_BIN:
            continue
        sub = d[m]
        y = sub['is_contact'].astype(float).to_numpy()
        rows.append(dict(
            axis=axis, binning=mode, bin=b, lo=float(edges[b]),
            hi=float(edges[b + 1]), mid=float(np.median(v[m])), n=int(m.sum()),
            base_rate=float(y.mean()),
            bce=bce(y, sub['p_contact']), skill=skill(y, sub['p_contact']),
            mean_timing_resid=float(
                sub['timing_residual_context'].mean(skipna=True)),
            mean_offset_resid=float(
                sub['offset_residual_context'].mean(skipna=True)),
        ))
    return pd.DataFrame(rows)


def plot_regions(prof, path):
    axes_list = AXES
    fig, axs = plt.subplots(2, len(axes_list), figsize=(5.0 * len(axes_list), 8),
                            squeeze=False, constrained_layout=True)
    for j, ax_name in enumerate(axes_list):
        for i, mode in enumerate(['quantile', 'width']):
            ax = axs[i][j]
            s = prof[(prof['axis'] == ax_name) &
                     (prof['binning'] == mode)].sort_values('mid')
            if s.empty:
                ax.set_visible(False)
                continue
            ax.plot(s['mid'], s['skill'], color=RED, marker='o', ms=4, lw=2,
                    label='skill')
            ax.set_ylabel('skill' if j == 0 else '')
            ax.grid(alpha=0.3)
            ax2 = ax.twinx()
            ax2.bar(s['mid'], s['n'], width=(s['mid'].diff().median() or 1) * 0.7,
                    color=GRAY, alpha=0.25)
            ax2.set_ylabel('bin n' if j == len(axes_list) - 1 else '')
            ax2.grid(False)
            r = (stats.spearmanr(s['skill'], s['n']).statistic
                 if len(s) > 3 else np.nan)
            ax.set_title(f'{ax_name} — {mode} bins\n'
                         f'skill vs n: rho={r:+.2f}', fontsize=10)
            if i == 1:
                ax.set_xlabel(ax_name)
    fig.suptitle('Direction 2 — contact-difficulty by pitch region, with bin '
                 'size overlaid (grey bars)\n'
                 'equal-count bins hold n fixed; equal-width bins let it vary',
                 fontsize=13, fontweight='bold')
    fig.savefig(path, dpi=135, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def plot_difficulty(ev, path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    d = ev['difficulty']
    axs[0].hist(d, bins=60, color=BLUE)
    axs[0].set_title('Per-pitch contact difficulty\n'
                     'P(whiff | swing, pitch characteristics)', fontsize=11)
    axs[0].set_xlabel('difficulty')
    axs[0].grid(alpha=0.3)

    by_pt = (ev.groupby('pitch_type')['difficulty']
             .agg(['mean', 'size']).query('size >= 2000')
             .sort_values('mean'))
    axs[1].barh(by_pt.index, by_pt['mean'], color=RED)
    axs[1].set_title('Mean difficulty by pitch type', fontsize=11)
    axs[1].grid(alpha=0.3, axis='x')

    s = ev.dropna(subset=['plate_z', 'pfx_z'])
    hb = axs[2].hexbin(s['pfx_z'], s['plate_z'], C=s['difficulty'],
                       gridsize=28, cmap='Reds', mincnt=50)
    axs[2].set_xlabel('pfx_z (vertical break)')
    axs[2].set_ylabel('plate_z (height)')
    axs[2].set_title('Difficulty surface', fontsize=11)
    fig.colorbar(hb, ax=axs[2], label='mean difficulty')
    fig.suptitle('Direction 2 — pitch difficulty score', fontsize=13,
                 fontweight='bold')
    fig.savefig(path, dpi=135, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


def main():
    ap = argparse.ArgumentParser(description='Direction 2: pitch grading')
    ap.add_argument('--min-pitches', type=int, default=300)
    args = ap.parse_args()

    tr = load_base(TRAIN_SEASON).rename(
        columns={'predicted_timing_oof': 'predicted_timing',
                 'predicted_offset_oof': 'predicted_offset'})
    ev = load_base(EVAL_SEASON).rename(
        columns={'predicted_timing_context': 'predicted_timing',
                 'predicted_offset_context': 'predicted_offset'})
    print(f'train {len(tr):,} swings | eval {len(ev):,} swings')
    ev = score_models(tr, ev)
    ev['difficulty'] = 1.0 - ev['p_contact_pitch']

    # 1 + 2: region profiling under both binning schemes
    prof = pd.concat([profile_axis(ev, a, m)
                      for a in AXES for m in ('quantile', 'width')],
                     ignore_index=True)
    prof.to_csv(os.path.join(OUT_DIR, 'region_profile.csv'), index=False)
    plot_regions(prof, os.path.join(PLOT_DIR, 'region_skill.png'))

    spars = []
    for a in AXES:
        for m in ('quantile', 'width'):
            s = prof[(prof['axis'] == a) & (prof['binning'] == m)]
            if len(s) < 4:
                continue
            spars.append(dict(
                axis=a, binning=m, n_bins=len(s),
                skill_min=float(s['skill'].min()),
                skill_max=float(s['skill'].max()),
                skill_range=float(s['skill'].max() - s['skill'].min()),
                n_min=int(s['n'].min()), n_max=int(s['n'].max()),
                rho_skill_vs_n=float(stats.spearmanr(s['skill'],
                                                     s['n']).statistic)))
    spars = pd.DataFrame(spars)
    spars.to_csv(os.path.join(OUT_DIR, 'sparsity_check.csv'), index=False)

    # 3: difficulty score summary
    summ = ev['difficulty'].describe(percentiles=[.05, .25, .5, .75, .95])
    pd.DataFrame({'difficulty': summ}).to_csv(
        os.path.join(OUT_DIR, 'pitch_difficulty.csv'))

    # 4: pitcher aggregates
    g = ev.groupby('pitcher')
    pit = pd.DataFrame({'n_swings': g.size(),
                        'difficulty': g['difficulty'].mean(),
                        'actual_whiff_pct': 1 - g['is_contact'].mean(),
                        'mean_velo': g['release_speed'].mean(),
                        'mean_pfx_z': g['pfx_z'].mean()})
    pit = pit[pit['n_swings'] >= args.min_pitches].copy()
    pit['difficulty_pctl'] = pit['difficulty'].rank(pct=True) * 100
    pit = pit.sort_values('difficulty', ascending=False)
    pit.to_csv(os.path.join(OUT_DIR, 'pitcher_grades.csv'))
    plot_difficulty(ev, os.path.join(PLOT_DIR, 'difficulty_dist.png'))

    pd.set_option('display.width', 210)
    print('\n=== Sparsity check (does skill track bin size?) ===\n')
    print(spars.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
    print('\n=== Skill across the flagged axes (equal-count bins) ===\n')
    for a in AXES:
        s = prof[(prof['axis'] == a) &
                 (prof['binning'] == 'quantile')].sort_values('mid')
        print(f'-- {a}: skill {s["skill"].iloc[0]:.3f} (low end) -> '
              f'{s["skill"].iloc[-1]:.3f} (high end), '
              f'n per bin {s["n"].min():,}-{s["n"].max():,}')
    print(f'\n=== Pitcher difficulty, top 12 (>= {args.min_pitches} swings) ===\n')
    print(pit.head(12).to_string(float_format=lambda v: f'{v:,.4f}'))
    print(f'\ncorr(difficulty, actual whiff%) across pitchers = '
          f'{pit["difficulty"].corr(pit["actual_whiff_pct"]):.3f}')
    print(f'-> {OUT_DIR}')


if __name__ == '__main__':
    main()
