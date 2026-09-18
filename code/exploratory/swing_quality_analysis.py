"""
swing_quality_analysis.py
==========================
Analyses the relationship between timing quality, barrel placement quality,
and swing outcomes at both the swing and batter level.

Core idea
---------
Each swing on which we have model predictions gets two residuals:

  timing_resid   = int_y_actual - int_y_predicted
                   (intercept_ball_minus_batter_pos_y_inches - int_y_predicted)
                   Bad in BOTH directions — being too early (negative) or too
                   late (positive) relative to prediction both indicate poor
                   timing. We use |timing_resid| to measure timing error.

  barrel_resid   = barrel_distance_v2 - barrel_pred_v3
                   Signed — negative means better than predicted (good),
                   positive means worse than predicted (bad).
                   We use raw barrel_resid for direction and barrel_resid > 0
                   as the "worse than expected" indicator.

Threshold classification (1 SD, computed on the full swing population)
-----------------------------------------------------------------------
  timing_bad    = |timing_resid| > timing_resid_sd
  barrel_bad    = barrel_resid   > barrel_resid_sd   (positive = worse)

Miss classification (swinging strikes only)
-------------------------------------------
  Each miss is classified into one of four buckets:
    timing_only   — timing_bad & ~barrel_bad
    barrel_only   — ~timing_bad & barrel_bad
    both          — timing_bad & barrel_bad
    neither       — ~timing_bad & ~barrel_bad

Analyses
--------
1. Swing-level: residual distributions overall and by event type
2. Swing-level: do timing-driven misses vs barrel-driven misses differ
   in delta_run_exp? (for misses where delta_run_exp is available)
3. Batter-level: what fraction of each batter's misses fall into each bucket?
4. Batter-level: do batters who miss more in one category have better
   outcomes? Correlated against xwOBA (from batter_stats_2025.csv).

Outputs (written to out/diagnostics/)
--------------------------------------
  swing_quality_residual_distributions.png
  swing_quality_miss_classification.png
  swing_quality_miss_vs_run_exp.png
  swing_quality_batter_buckets.png
  swing_quality_bucket_vs_xwoba.png
  swing_quality_batter_summary.csv
  swing_quality_miss_summary.csv

Requires: pandas, numpy, matplotlib, seaborn, scipy
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'all_pitches_2025')
DIAG_DIR = os.path.join(BASE_DIR, 'out', 'exploratory', 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

BATTER_STATS_PATH = os.path.join(BASE_DIR, 'data', 'batter_stats_2025.csv')

# ── Column names ───────────────────────────────────────────────────────────────
INT_Y_RAW_COL   = 'intercept_ball_minus_batter_pos_y_inches'
INT_Y_PRED_COL  = 'int_y_predicted'
BARREL_ACT_COL  = 'barrel_distance_v2'
BARREL_PRED_COL = 'barrel_pred_v3'
RUN_EXP_COL     = 'delta_run_exp'

MISS    = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL    = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS

# Classification threshold: n SDs from mean residual
THRESHOLD_SD = 1.0

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE   = '#2563EB'; RED  = '#DC2626'; GREEN  = '#16A34A'
GRAY   = '#6B7280'; PURPLE = '#7C3AED'; ORANGE = '#D97706'

BUCKET_COLORS = {
    'timing_only':  ORANGE,
    'barrel_only':  BLUE,
    'both':         RED,
    'neither':      GRAY,
}
BUCKET_LABELS = {
    'timing_only': 'Timing-driven miss',
    'barrel_only': 'Barrel-driven miss',
    'both':        'Both (timing + barrel)',
    'neither':     'Neither (unexplained)',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_swings() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # Rename int_y raw column
    if INT_Y_RAW_COL in df.columns:
        df = df.rename(columns={INT_Y_RAW_COL: 'int_y_actual'})
    elif 'int_y' in df.columns:
        df = df.rename(columns={'int_y': 'int_y_actual'})
    else:
        raise ValueError(
            f'Cannot find int_y column. Expected {INT_Y_RAW_COL!r} or int_y.')

    # Verify prediction columns exist
    for col in [INT_Y_PRED_COL, BARREL_ACT_COL, BARREL_PRED_COL]:
        if col not in df.columns:
            raise ValueError(
                f'Required column {col!r} not found. '
                f'Run run_frozen_models.py and write_barrel_swing_predictions.py first.')

    df['batter'] = df['batter'].astype(str)

    # Filter to swings only
    swings = df[df['description'].isin(ALL_SWINGS)].copy()
    print(f'Swing events: {len(swings):,}')
    return swings


def load_batter_stats() -> pd.DataFrame | None:
    if not os.path.exists(BATTER_STATS_PATH):
        print(f'  Batter stats not found at {BATTER_STATS_PATH} — '
              f'xwOBA analyses will be skipped')
        return None
    stats = pd.read_csv(BATTER_STATS_PATH)
    stats['batter_id'] = stats['batter_id'].astype(str)

    # Find xwOBA column
    xwoba_col = next((c for c in ['xst_woba', 'xst_xwoba', 'xwoba', 'xwOBA']
                      if c in stats.columns), None)
    if xwoba_col is None:
        print(f'  xwOBA column not found in batter stats. '
              f'woba-like cols: {[c for c in stats.columns if "woba" in c.lower()]}')
        return None

    stats = stats[['batter_id', xwoba_col]].rename(
        columns={xwoba_col: 'xwoba'})
    print(f'  Batter stats: {len(stats):,} rows with xwOBA')
    return stats.dropna(subset=['xwoba'])


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE RESIDUALS AND CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(swings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute timing and barrel residuals where predictions are available.

    timing_resid = int_y_actual - int_y_predicted  (bad in both directions)
    barrel_resid = barrel_distance_v2 - barrel_pred_v3  (negative = good)

    Both require their respective prediction columns to be non-null.
    """
    df = swings.copy()

    # Timing residual
    has_timing = df['int_y_actual'].notna() & df[INT_Y_PRED_COL].notna()
    df['timing_resid'] = np.where(
        has_timing,
        df['int_y_actual'] - df[INT_Y_PRED_COL],
        np.nan
    )
    df['timing_abs_resid'] = df['timing_resid'].abs()

    # Barrel residual
    has_barrel = df[BARREL_ACT_COL].notna() & df[BARREL_PRED_COL].notna()
    df['barrel_resid'] = np.where(
        has_barrel,
        df[BARREL_ACT_COL] - df[BARREL_PRED_COL],
        np.nan
    )

    n_timing = has_timing.sum()
    n_barrel = has_barrel.sum()
    print(f'\nResiduals computed:')
    print(f'  timing: {n_timing:,} swings  '
          f'(mean={df["timing_resid"].mean():.3f}, '
          f'sd={df["timing_resid"].std():.3f})')
    print(f'  barrel: {n_barrel:,} swings  '
          f'(mean={df["barrel_resid"].mean():.3f}, '
          f'sd={df["barrel_resid"].std():.3f})')

    return df


def classify_swings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Classify each swing as timing_bad / barrel_bad based on 1 SD thresholds.

    timing_bad  = |timing_resid| > THRESHOLD_SD * sd(|timing_resid|)
    barrel_bad  = barrel_resid   > THRESHOLD_SD * sd(barrel_resid)
                  (positive barrel_resid = worse than predicted)

    Thresholds are computed on the full swing population (not just misses)
    so they represent "worse than a typical swing" rather than
    "worse than a typical miss."

    Returns (df_classified, thresholds_dict).
    """
    timing_sd = df['timing_abs_resid'].std()
    barrel_sd = df['barrel_resid'].std()

    timing_thresh = THRESHOLD_SD * timing_sd * 0.5
    barrel_thresh = THRESHOLD_SD * barrel_sd * 0.5

    thresholds = {
        'timing_sd': timing_sd,
        'barrel_sd': barrel_sd,
        'timing_thresh': timing_thresh,
        'barrel_thresh': barrel_thresh,
    }

    print(f'\nClassification thresholds ({THRESHOLD_SD} SD):')
    print(f'  timing: |resid| > {timing_thresh:.3f} inches  '
          f'(SD={timing_sd:.3f})')
    print(f'  barrel: resid   > {barrel_thresh:.3f} inches  '
          f'(SD={barrel_sd:.3f})')

    df = df.copy()
    df['timing_bad'] = df['timing_abs_resid'] > timing_thresh
    df['barrel_bad'] = df['barrel_resid']      > barrel_thresh

    # Four-way classification
    conditions = [
        df['timing_bad'] & ~df['barrel_bad'],
        ~df['timing_bad'] & df['barrel_bad'],
        df['timing_bad'] & df['barrel_bad'],
        ~df['timing_bad'] & ~df['barrel_bad'],
    ]
    choices = ['timing_only', 'barrel_only', 'both', 'neither']
    df['miss_bucket'] = np.select(conditions, choices, default=None)

    # Only meaningful on swings where both residuals are available
    no_pred = df['timing_resid'].isna() | df['barrel_resid'].isna()
    df.loc[no_pred, 'miss_bucket'] = np.nan

    return df, thresholds


# ══════════════════════════════════════════════════════════════════════════════
# 3. SWING-LEVEL ANALYSES
# ══════════════════════════════════════════════════════════════════════════════

def plot_residual_distributions(df: pd.DataFrame, thresholds: dict):
    """
    Panel 1: timing_resid distribution (all swings with prediction)
    Panel 2: barrel_resid distribution (all swings with prediction)
    Threshold lines shown on each panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Residual distributions across all swing events',
                 fontsize=13, fontweight='bold')

    # Timing
    ax = axes[0]
    tr = df['timing_resid'].dropna()
    lo, hi = tr.quantile(0.005), tr.quantile(0.995)
    ax.hist(tr.clip(lo, hi), bins=80, color=BLUE,
            edgecolor='white', alpha=0.85)
    t = thresholds['timing_thresh']
    ax.axvline( t, color=RED, linewidth=1.5, linestyle='--',
               label=f'+{THRESHOLD_SD}SD = +{t:.2f}"')
    ax.axvline(-t, color=RED, linewidth=1.5, linestyle='--',
               label=f'-{THRESHOLD_SD}SD = -{t:.2f}"')
    ax.axvline(0, color=GRAY, linewidth=1.0, linestyle=':')
    ax.set_xlabel('Timing residual (inches)\nactual − predicted int_y')
    ax.set_ylabel('Count')
    ax.set_title('Timing residual\n(bad in both directions)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Barrel
    ax = axes[1]
    br = df['barrel_resid'].dropna()
    lo, hi = br.quantile(0.005), br.quantile(0.995)
    ax.hist(br.clip(lo, hi), bins=80, color=GREEN,
            edgecolor='white', alpha=0.85)
    b = thresholds['barrel_thresh']
    ax.axvline(b, color=RED, linewidth=1.5, linestyle='--',
               label=f'+{THRESHOLD_SD}SD = +{b:.2f}" (worse than predicted)')
    ax.axvline(0, color=GRAY, linewidth=1.0, linestyle=':',
               label='0 (as predicted)')
    ax.set_xlabel('Barrel residual (inches)\nactual − predicted barrel_distance_v2')
    ax.set_ylabel('Count')
    ax.set_title('Barrel residual\n(negative = better than predicted)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(DIAG_DIR, 'swing_quality_residual_distributions.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_miss_classification(df: pd.DataFrame):
    """
    Bar chart: how many misses fall into each bucket?
    Broken down by swing type (swinging_strike vs missed_bunt etc.)
    """
    misses = df[df['description'].isin(MISS) & df['miss_bucket'].notna()].copy()
    print(f'\nMiss classification ({len(misses):,} misses with both predictions):')

    counts = misses['miss_bucket'].value_counts()
    pcts   = (counts / len(misses) * 100).round(1)
    for bucket in ['timing_only', 'barrel_only', 'both', 'neither']:
        n = counts.get(bucket, 0)
        p = pcts.get(bucket, 0)
        print(f'  {BUCKET_LABELS[bucket]}: {n:,} ({p}%)')

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(4)
    order = ['timing_only', 'barrel_only', 'both', 'neither']
    vals  = [counts.get(b, 0) for b in order]
    bars  = ax.bar(x, vals,
                   color=[BUCKET_COLORS[b] for b in order],
                   edgecolor='white', alpha=0.85)
    for bar, v, p in zip(bars, vals, [pcts.get(b, 0) for b in order]):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + max(vals)*0.01,
                f'{v:,}\n({p}%)', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b] for b in order],
                       fontsize=10)
    ax.set_ylabel('Number of misses')
    ax.set_title('Miss classification: timing vs barrel vs both vs neither',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(DIAG_DIR, 'swing_quality_miss_classification.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    return misses


def plot_miss_vs_run_exp(misses: pd.DataFrame):
    """
    Box + strip plot: delta_run_exp by miss bucket.
    Tests whether timing-driven misses differ in run value from
    barrel-driven misses.
    Also reports one-way ANOVA and pairwise t-tests.
    """
    if RUN_EXP_COL not in misses.columns:
        print(f'  {RUN_EXP_COL!r} not in data — swing-level run value '
              f'analysis skipped')
        return

    sub = misses[misses[RUN_EXP_COL].notna() &
                 misses['miss_bucket'].notna()].copy()
    if len(sub) < 20:
        print(f'  Only {len(sub)} rows with {RUN_EXP_COL} — skipping')
        return

    print(f'\nSwing-level run value by miss bucket '
          f'(n={len(sub):,} misses with delta_run_exp):')
    order = ['timing_only', 'barrel_only', 'both', 'neither']
    for bucket in order:
        grp = sub[sub['miss_bucket'] == bucket][RUN_EXP_COL]
        if len(grp) > 0:
            print(f'  {BUCKET_LABELS[bucket]}: '
                  f'mean={grp.mean():.4f}  sd={grp.std():.4f}  '
                  f'n={len(grp):,}')

    # ANOVA
    groups = [sub[sub['miss_bucket'] == b][RUN_EXP_COL].dropna()
              for b in order if (sub['miss_bucket'] == b).sum() > 1]
    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f'  One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}')

    # Pairwise t-test: timing_only vs barrel_only
    t_grp = sub[sub['miss_bucket'] == 'timing_only'][RUN_EXP_COL].dropna()
    b_grp = sub[sub['miss_bucket'] == 'barrel_only'][RUN_EXP_COL].dropna()
    if len(t_grp) > 1 and len(b_grp) > 1:
        t_stat, p_t = stats.ttest_ind(t_grp, b_grp)
        print(f'  t-test (timing_only vs barrel_only): '
              f't={t_stat:.3f}, p={p_t:.4f}')

    fig, ax = plt.subplots(figsize=(11, 6))
    valid_order = [b for b in order
                   if (sub['miss_bucket'] == b).sum() > 0]

    sns.stripplot(data=sub, x='miss_bucket', y=RUN_EXP_COL,
                  order=valid_order, ax=ax,
                  palette=BUCKET_COLORS, alpha=0.15, size=2,
                  jitter=True, zorder=1)
    sns.boxplot(data=sub, x='miss_bucket', y=RUN_EXP_COL,
                order=valid_order, ax=ax,
                palette={b: 'white' for b in valid_order},
                width=0.5, linewidth=1.2, fliersize=0,
                boxprops=dict(edgecolor=GRAY),
                medianprops=dict(color=RED, linewidth=2.0),
                whiskerprops=dict(color=GRAY),
                capprops=dict(color=GRAY),
                zorder=2)
    ax.axhline(0, color=RED, linewidth=1.0, linestyle='--')
    ax.set_xticklabels([BUCKET_LABELS[b] for b in valid_order], fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('Delta run expectancy', fontsize=11)
    ax.set_title('Run value by miss type\n'
                 '(do timing-driven misses differ from barrel-driven?)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(DIAG_DIR, 'swing_quality_miss_vs_run_exp.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 4. BATTER-LEVEL ANALYSES
# ══════════════════════════════════════════════════════════════════════════════

def build_batter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each batter, compute:
      - mean and SD of timing_abs_resid and barrel_resid (all swings)
      - fraction of swings that are timing_bad / barrel_bad
      - miss bucket fractions (on misses only)
      - mean delta_run_exp by bucket (if available)
    """
    swings_with_pred = df[df['timing_resid'].notna() &
                          df['barrel_resid'].notna()].copy()
    misses = swings_with_pred[swings_with_pred['description'].isin(MISS)]

    # ── All-swing stats ────────────────────────────────────────────────────────
    swing_agg = (swings_with_pred
                 .groupby('batter')
                 .agg(
                     n_swings_pred      = ('timing_resid', 'size'),
                     mean_timing_abs    = ('timing_abs_resid', 'mean'),
                     mean_barrel_resid  = ('barrel_resid', 'mean'),
                     pct_timing_bad     = ('timing_bad', 'mean'),
                     pct_barrel_bad     = ('barrel_bad', 'mean'),
                 )
                 .reset_index())

    # ── Miss bucket fractions ──────────────────────────────────────────────────
    miss_agg = (misses
                .groupby('batter')
                .agg(n_misses=('miss_bucket', 'size'))
                .reset_index())

    for bucket in ['timing_only', 'barrel_only', 'both', 'neither']:
        bucket_counts = (misses[misses['miss_bucket'] == bucket]
                         .groupby('batter').size()
                         .rename(f'n_{bucket}')
                         .reset_index())
        miss_agg = miss_agg.merge(bucket_counts, on='batter', how='left')
        miss_agg[f'n_{bucket}'] = miss_agg[f'n_{bucket}'].fillna(0).astype(int)
        miss_agg[f'pct_{bucket}'] = (miss_agg[f'n_{bucket}'] /
                                      miss_agg['n_misses'])

    # ── Run value by bucket ────────────────────────────────────────────────────
    if RUN_EXP_COL in misses.columns:
        for bucket in ['timing_only', 'barrel_only', 'both', 'neither']:
            re_agg = (misses[misses['miss_bucket'] == bucket]
                      .groupby('batter')[RUN_EXP_COL]
                      .mean()
                      .rename(f'mean_re_{bucket}')
                      .reset_index())
            miss_agg = miss_agg.merge(re_agg, on='batter', how='left')

    summary = swing_agg.merge(miss_agg, on='batter', how='left')
    print(f'\nBatter summary: {len(summary):,} batters')
    return summary


def plot_batter_buckets(summary: pd.DataFrame):
    """
    Stacked bar chart of mean miss bucket fractions across all batters,
    sorted by pct_timing_only to show the range of timing vs barrel profiles.
    Only show batters with at least 20 classified misses for stability.
    """
    sub = summary[summary['n_misses'] >= 20].sort_values(
        'pct_timing_only', ascending=False).copy()

    if len(sub) < 5:
        print('  Fewer than 5 batters with 20+ misses — skipping bucket plot')
        return

    fig, ax = plt.subplots(figsize=(max(12, len(sub)*0.18), 6))

    x      = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    order  = ['timing_only', 'barrel_only', 'both', 'neither']

    for bucket in order:
        col = f'pct_{bucket}'
        vals = sub[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom,
               color=BUCKET_COLORS[bucket],
               label=BUCKET_LABELS[bucket],
               alpha=0.85, edgecolor='none', width=1.0)
        bottom += vals

    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel('Fraction of classified misses')
    ax.set_title('Miss bucket profile per batter\n'
                 '(sorted by timing-only fraction, ≥20 misses)',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(DIAG_DIR, 'swing_quality_batter_buckets.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_bucket_vs_xwoba(summary: pd.DataFrame,
                          batter_stats: pd.DataFrame | None):
    """
    2x2 scatter: pct_timing_only, pct_barrel_only, pct_both, pct_neither
    each vs xwOBA. Tests whether batters who miss in one way more often
    have systematically better or worse outcomes.
    """
    if batter_stats is None:
        print('  Skipping bucket vs xwOBA — no batter stats')
        return

    merged = summary.merge(batter_stats,
                           left_on='batter', right_on='batter_id',
                           how='inner')
    merged = merged[merged['n_misses'] >= 20].copy()
    print(f'\nBatter-level xwOBA analysis: {len(merged):,} batters '
          f'(≥20 misses + xwOBA available)')

    if len(merged) < 10:
        print('  Too few batters — skipping')
        return

    order   = ['timing_only', 'barrel_only', 'both', 'neither']
    titles  = [BUCKET_LABELS[b] for b in order]
    colors  = [BUCKET_COLORS[b] for b in order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle('Miss bucket fraction vs xwOBA (2025)\n'
                 'Do batters who miss more in one category have better outcomes?',
                 fontsize=13, fontweight='bold')

    for ax, bucket, title, color in zip(axes.flat, order, titles, colors):
        col  = f'pct_{bucket}'
        x    = merged[col]
        y    = merged['xwoba']
        mask = x.notna() & y.notna()

        if mask.sum() < 5:
            ax.set_title(title, fontweight='bold')
            ax.text(0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        xv, yv = x[mask].values, y[mask].values
        r, p   = stats.pearsonr(xv, yv)

        slope, intercept, *_ = stats.linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        y_line = slope * x_line + intercept

        ax.scatter(xv * 100, yv, s=14, alpha=0.5,
                   color=color, rasterized=True)
        ax.plot(x_line * 100, y_line, color=RED, linewidth=1.5,
                label=f'r={r:.3f}, p={p:.3f}, n={mask.sum()}')
        ax.set_xlabel(f'% of misses: {title}', fontsize=10)
        ax.set_ylabel('xwOBA 2025', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        print(f'  {title}: r={r:.3f}, p={p:.4f}, n={mask.sum()}')

    plt.tight_layout()
    path = os.path.join(DIAG_DIR, 'swing_quality_bucket_vs_xwoba.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Swing Quality Analysis')
    print('='*60)

    # ── Load ──────────────────────────────────────────────────────────────────
    swings       = load_swings()
    batter_stats = load_batter_stats()

    # ── Residuals and classification ───────────────────────────────────────────
    swings              = compute_residuals(swings)
    swings, thresholds  = classify_swings(swings)

    # ── Swing-level analyses ───────────────────────────────────────────────────
    print('\n── Swing-level analyses ──')
    plot_residual_distributions(swings, thresholds)
    misses = plot_miss_classification(swings)
    plot_miss_vs_run_exp(misses)

    # ── Batter-level analyses ──────────────────────────────────────────────────
    print('\n── Batter-level analyses ──')
    summary = build_batter_summary(swings)

    path = os.path.join(DIAG_DIR, 'swing_quality_batter_summary.csv')
    summary.to_csv(path, index=False)
    print(f'  Batter summary → {path}')

    miss_summary = (swings[swings['description'].isin(MISS) &
                            swings['miss_bucket'].notna()]
                    [['batter', 'description', 'miss_bucket',
                      'timing_resid', 'timing_abs_resid', 'barrel_resid']
                     + ([RUN_EXP_COL] if RUN_EXP_COL in swings.columns else [])]
                    .copy())
    path = os.path.join(DIAG_DIR, 'swing_quality_miss_summary.csv')
    miss_summary.to_csv(path, index=False)
    print(f'  Miss summary → {path}')

    plot_batter_buckets(summary)
    plot_bucket_vs_xwoba(summary, batter_stats)

    print(f'\nAll outputs in {DIAG_DIR}')


if __name__ == '__main__':
    main()