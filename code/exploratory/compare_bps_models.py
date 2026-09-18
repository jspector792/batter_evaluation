"""
swing_model_collapse_diagnostics.py
====================================
Investigates why the timing model (int_y) produces a singular random-effects
covariance while the tilt model (swing_path_tilt) does not, despite both using
the same batter/pitch_group grouping structure.

Tests four hypotheses in sequence:

  H1. The fixed effects absorb all between-group variance in int_y but not
      swing_path_tilt — leaving nothing for the random intercept to capture.

  H2. The ICC (intraclass correlation) is genuinely near zero for int_y,
      meaning batters simply don't differ systematically in contact depth
      after controlling for pitch characteristics.

  H3. Within-group noise swamps between-group signal for int_y — the
      between/within variance ratio is too low for the optimizer to detect.

  H4. The distribution of group means is qualitatively different between the
      two outcomes — int_y group means cluster tightly around the grand mean
      while swing_path_tilt group means are widely dispersed.

For each hypothesis, the script:
  - Computes relevant statistics for both outcomes side-by-side
  - Produces a labelled diagnostic plot
  - Prints a plain-English interpretation

All plots are saved to out/diagnostics/collapse/.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "diagnostics", "collapse")
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
IN_PLAY     = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL        = {'foul', 'foul_tip', 'foul_bunt'}
MISS        = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT     = IN_PLAY | FOUL
ALL_SWINGS  = CONTACT | MISS
N_CLUSTERS  = 10

sns.set_theme(style='whitegrid', font_scale=1.1)
BLUE  = '#2563EB'
GREEN = '#16A34A'
RED   = '#DC2626'
GRAY  = '#9CA3AF'
DARK  = '#1F2937'


# ══════════════════════════════════════════════════════════════════════════════
# 0.  LOAD + CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df['pfx_x_arm'] = df['pfx_x'] * df['p_throws'].map({'R': 1, 'L': -1}).fillna(1)
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    feats = ['pfx_x_arm', 'pfx_z', 'release_speed']
    df    = df.dropna(subset=feats + ['pitch_type', 'p_throws']).copy()

    scaler = StandardScaler().fit(df[feats].values)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(
                scaler.transform(df[feats].values))
    df['pitch_cluster'] = km.labels_.astype(str)

    rs_mean = df['release_speed'].mean()
    df['release_speed_c'] = df['release_speed'] - rs_mean

    df['_group'] = df['batter'].astype(str) + '__' + df['pitch_cluster'].astype(str)
    return df


def build_datasets(df):
    """
    Return the two model-specific datasets with consistent column naming.
    timing : all swings,  outcome = int_y
    tilt   : misses only, outcome = swing_path_tilt
    Both have: outcome, group (_group), batter, pitch_cluster, fixed-effect vars
    """
    # ── Timing ────────────────────────────────────────────────────────────────
    t_needed = [INTERCEPT_Y, 'release_speed_c', 'plate_x_bat_flip',
                'batter', 'pitch_cluster', '_group', 'description']
    df_t = (df[df['description'].isin(ALL_SWINGS)]
            .dropna(subset=t_needed)
            .rename(columns={INTERCEPT_Y: 'outcome'})
            .copy())
    df_t['model']    = 'timing (int_y)'
    df_t['fe_terms'] = 'release_speed_c + plate_x_bat_flip'

    # ── Tilt ──────────────────────────────────────────────────────────────────
    s_needed = ['swing_path_tilt', 'plate_z', 'batter',
                'pitch_cluster', '_group', 'description']
    df_s = (df[df['description'].isin(MISS)]
            .dropna(subset=s_needed)
            .rename(columns={'swing_path_tilt': 'outcome'})
            .copy())
    df_s['model']    = 'tilt (swing_path_tilt)'
    df_s['fe_terms'] = 'plate_z'

    print(f'Timing dataset : {len(df_t):,} rows, '
          f'{df_t["_group"].nunique():,} groups')
    print(f'Tilt dataset   : {len(df_s):,} rows, '
          f'{df_s["_group"].nunique():,} groups')
    return df_t, df_s


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: null-model ICC
# ══════════════════════════════════════════════════════════════════════════════

def null_model_icc(df, group_col='_group'):
    """
    Fit outcome ~ 1 + (1|group) and return ICC = group_var / (group_var + resid_var).
    Uses a one-way ANOVA decomposition which is numerically stable and does
    not suffer the singular-matrix problem (it is just moment estimation).

    Returns
    -------
    icc         : float
    group_var   : float (between-group variance component)
    resid_var   : float (within-group / residual variance)
    group_means : pd.Series of per-group means
    """
    grand_mean  = df['outcome'].mean()
    group_means = df.groupby(group_col)['outcome'].mean()
    group_sizes = df.groupby(group_col)['outcome'].count()

    # Between-group sum of squares
    ss_between = ((group_means - grand_mean) ** 2 * group_sizes).sum()
    df_between = len(group_means) - 1

    # Within-group sum of squares
    within     = df.groupby(group_col)['outcome'].apply(
                     lambda x: ((x - x.mean()) ** 2).sum())
    ss_within  = within.sum()
    df_within  = len(df) - len(group_means)

    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within

    # Method-of-moments estimate of n_0 (harmonic mean of group sizes)
    n0 = (len(df) - (group_sizes ** 2).sum() / len(df)) / (len(group_means) - 1)

    resid_var = ms_within
    group_var = max(0.0, (ms_between - ms_within) / n0)
    total_var = group_var + resid_var
    icc       = group_var / total_var if total_var > 0 else 0.0

    return icc, group_var, resid_var, group_means


def residual_icc(df, fe_formula, group_col='_group'):
    """
    Compute ICC on OLS residuals after removing fixed effects.
    This answers: after plate_z / release_speed explain what they can,
    is there still between-group structure left?
    """
    ols     = smf.ols(f'outcome ~ {fe_formula}', data=df).fit()
    df      = df.copy()
    df['outcome'] = ols.resid   # replace outcome with residuals
    icc, gv, rv, gm = null_model_icc(df, group_col)
    return icc, gv, rv, ols.rsquared


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def diag_h1_fixed_effects_absorption(df_t, df_s):
    """
    H1: Fixed effects absorb between-group variance.

    Compare:
      - R² of fixed-effects-only OLS for each model
      - ICC before and after removing fixed effects
    If timing model OLS R² ≈ total between-group variance, H1 is supported.
    """
    print('\n' + '='*70)
    print('H1: DO FIXED EFFECTS ABSORB BETWEEN-GROUP VARIANCE?')
    print('='*70)

    results = {}
    for label, df, fe in [
        ('timing', df_t, 'release_speed_c + plate_x_bat_flip'),
        ('tilt',   df_s, 'plate_z'),
    ]:
        icc_raw, gv_raw, rv_raw, gm_raw = null_model_icc(df)
        icc_res, gv_res, rv_res, ols_r2 = residual_icc(df, fe)

        results[label] = dict(
            icc_raw=icc_raw, gv_raw=gv_raw, rv_raw=rv_raw,
            icc_res=icc_res, gv_res=gv_res, rv_res=rv_res,
            ols_r2=ols_r2,
        )
        print(f'\n  {label}:')
        print(f'    Raw ICC (no fixed effects) : {icc_raw:.4f}  '
              f'(group_var={gv_raw:.3f}, resid_var={rv_raw:.3f})')
        print(f'    OLS R² (fixed effects only): {ols_r2:.4f}')
        print(f'    Residual ICC (after FE)    : {icc_res:.4f}  '
              f'(group_var={gv_res:.3f}, resid_var={rv_res:.3f})')
        pct_absorbed = (1 - gv_res / gv_raw) * 100 if gv_raw > 0 else 100
        print(f'    Between-group var absorbed : {pct_absorbed:.1f}%')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (label, color) in zip(axes, [('timing', BLUE), ('tilt', GREEN)]):
        r  = results[label]
        categories = ['Raw\nbetween-group var', 'Residual\nbetween-group var',
                      'Within-group\n(residual) var']
        values     = [r['gv_raw'], r['gv_res'], r['rv_res']]
        alphas     = [1.0, 0.55, 0.3]
        bar_colors = [color, color, GRAY]

        bars = []
        for i, (cat, val, col, alp) in enumerate(
                zip(categories, values, bar_colors, alphas)):
            b = ax.bar(i, val, color=col, alpha=alp,
                       edgecolor='white', width=0.5)
            bars.append(b[0])

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(values) * 0.01,
                    f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9)

        ax.set_title(
            f'{label.capitalize()} model\n'
            f'OLS R²={r["ols_r2"]:.3f}  |  '
            f'Raw ICC={r["icc_raw"]:.4f}  |  '
            f'Residual ICC={r["icc_res"]:.4f}',
            fontsize=11, fontweight='bold'
        )
        ax.set_ylabel('Variance (units²)', fontsize=10)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle(
        'H1: Fixed-effects absorption of between-group variance\n'
        'Dark bar = raw between-group variance before fixed effects\n'
        'Mid bar  = between-group variance remaining after fixed effects\n'
        'Light bar = within-group (residual) variance after fixed effects',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    _save(fig, 'h1_fixed_effects_absorption.png')

    return results


def diag_h2_icc_distribution(df_t, df_s):
    """
    H2: ICC is genuinely near zero for int_y.

    Compute ICC at multiple grouping levels:
      - batter only
      - pitch_cluster only
      - batter × pitch_cluster (nested)
    and show the breakdown for both outcomes.
    """
    print('\n' + '='*70)
    print('H2: ICC AT MULTIPLE GROUPING LEVELS')
    print('='*70)

    levels = [
        ('batter',        'batter'),
        ('pitch_cluster', 'pitch_cluster'),
        ('batter×cluster','_group'),
    ]

    rows = []
    for label, df in [('timing (int_y)', df_t),
                       ('tilt (swing_path_tilt)', df_s)]:
        for level_name, group_col in levels:
            if group_col not in df.columns:
                continue
            icc, gv, rv, _ = null_model_icc(df, group_col)
            rows.append(dict(model=label, level=level_name,
                             icc=icc, group_var=gv, resid_var=rv))
            print(f'  {label:30s}  level={level_name:15s}  '
                  f'ICC={icc:.4f}  group_var={gv:.3f}')

    df_icc = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, metric, ylabel in zip(
        axes,
        ['icc', 'group_var'],
        ['ICC  (fraction of total variance)', 'Between-group variance component'],
    ):
        pivot = df_icc.pivot(index='level', columns='model', values=metric)
        x     = np.arange(len(pivot))
        w     = 0.35
        for i, (col, color) in enumerate(
            zip(pivot.columns, [BLUE, GREEN])
        ):
            bars = ax.bar(x + (i - 0.5) * w, pivot[col], w,
                          label=col, color=color, alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, pivot[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + max(pivot.values.flatten()) * 0.01,
                        f'{v:.4f}' if metric == 'icc' else f'{v:.2f}',
                        ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle('H2: ICC at multiple grouping levels\n'
                 'If ICC ≈ 0 at batter×cluster, the mixed model has nothing to estimate',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, 'h2_icc_by_level.png')

    return df_icc


def diag_h3_within_between_ratio(df_t, df_s):
    """
    H3: Within-group noise swamps between-group signal.

    For each group, compute:
      - group mean deviation from grand mean  (between-group signal)
      - within-group standard deviation       (within-group noise)
      - signal-to-noise ratio: |group_mean_dev| / within_std

    Plot the distribution of SNR across groups for both models.
    A model where most groups have SNR << 1 will collapse.
    """
    print('\n' + '='*70)
    print('H3: WITHIN-GROUP NOISE vs BETWEEN-GROUP SIGNAL')
    print('='*70)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for row_idx, (label, df, color) in enumerate([
        ('timing (int_y)', df_t, BLUE),
        ('tilt (swing_path_tilt)', df_s, GREEN),
    ]):
        grand_mean = df['outcome'].mean()
        grp        = df.groupby('_group')['outcome']
        gstats     = pd.DataFrame({
            'mean'    : grp.mean(),
            'std'     : grp.std(),
            'n'       : grp.count(),
        })
        gstats['mean_dev'] = (gstats['mean'] - grand_mean).abs()
        # SNR: how many within-group SDs does the group mean deviate?
        gstats['snr']      = gstats['mean_dev'] / (gstats['std'] + 1e-9)
        gstats             = gstats[gstats['n'] >= 5]   # exclude singletons

        pct_snr_gt1 = (gstats['snr'] > 1).mean() * 100
        median_snr  = gstats['snr'].median()
        print(f'\n  {label}:')
        print(f'    median SNR across groups  : {median_snr:.4f}')
        print(f'    % groups with SNR > 1     : {pct_snr_gt1:.1f}%')
        print(f'    grand mean                : {grand_mean:.3f}')
        print(f'    mean of group mean devs   : {gstats["mean_dev"].mean():.3f}')
        print(f'    mean within-group std     : {gstats["std"].mean():.3f}')

        # Panel 1: distribution of group mean deviations
        ax = axes[row_idx, 0]
        ax.hist(gstats['mean_dev'], bins=60, color=color,
                edgecolor='white', alpha=0.85)
        ax.axvline(gstats['mean_dev'].mean(), color=RED,
                   linewidth=1.5, linestyle='--',
                   label=f'mean={gstats["mean_dev"].mean():.3f}')
        ax.set_xlabel('|group mean − grand mean|', fontsize=9)
        ax.set_ylabel('Number of groups', fontsize=9)
        ax.set_title(f'{label}\nBetween-group signal', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: distribution of within-group SDs
        ax = axes[row_idx, 1]
        ax.hist(gstats['std'], bins=60, color=color,
                edgecolor='white', alpha=0.85)
        ax.axvline(gstats['std'].mean(), color=RED,
                   linewidth=1.5, linestyle='--',
                   label=f'mean={gstats["std"].mean():.3f}')
        ax.set_xlabel('Within-group standard deviation', fontsize=9)
        ax.set_ylabel('Number of groups', fontsize=9)
        ax.set_title(f'{label}\nWithin-group noise', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3: SNR distribution
        ax = axes[row_idx, 2]
        snr_clip = gstats['snr'].clip(upper=gstats['snr'].quantile(0.99))
        ax.hist(snr_clip, bins=60, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(1.0, color=RED, linewidth=1.5, linestyle='--',
                   label='SNR = 1')
        ax.axvline(median_snr, color=DARK, linewidth=1.5, linestyle=':',
                   label=f'median={median_snr:.3f}')
        ax.set_xlabel('SNR = |group mean dev| / within-group SD', fontsize=9)
        ax.set_ylabel('Number of groups', fontsize=9)
        ax.set_title(f'{label}\nSNR  ({pct_snr_gt1:.1f}% of groups > 1)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'H3: Within-group noise vs between-group signal\n'
        'Mixed model needs SNR >> 1 to estimate non-zero group variance\n'
        'Left = signal  |  Centre = noise  |  Right = SNR',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save(fig, 'h3_snr_analysis.png')


def diag_h4_group_mean_distributions(df_t, df_s):
    """
    H4: Distribution of group means differs qualitatively.

    If int_y group means cluster tightly around the grand mean (low spread),
    the model correctly finds Group Var ≈ 0.
    If swing_path_tilt group means are widely dispersed, Group Var is real.

    Also plots: group mean vs group size (does sparsity drive the pattern?),
    and Q-Q plots of group means vs normal distribution.
    """
    print('\n' + '='*70)
    print('H4: DISTRIBUTION OF GROUP MEANS')
    print('='*70)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for row_idx, (label, df, color) in enumerate([
        ('timing (int_y)', df_t, BLUE),
        ('tilt (swing_path_tilt)', df_s, GREEN),
    ]):
        grand_mean  = df['outcome'].mean()
        grp         = df.groupby('_group')['outcome']
        group_means = grp.mean()
        group_sizes = grp.count()
        group_stds  = grp.std()

        # Standardised group means (z-score relative to grand mean and
        # between-group SD): if normally distributed and well-spread,
        # this should look N(0,1). If collapsed, it looks like a spike.
        gm_std = group_means.std()
        z_means = (group_means - grand_mean) / gm_std if gm_std > 0 else group_means * 0

        print(f'\n  {label}:')
        print(f'    Grand mean                 : {grand_mean:.3f}')
        print(f'    SD of group means          : {gm_std:.4f}')
        print(f'    SD of raw outcome          : {df["outcome"].std():.4f}')
        print(f'    Fraction explained by groups: '
              f'{(gm_std**2 / df["outcome"].var()):.4f}')

        # Panel 1: histogram of group means
        ax = axes[row_idx, 0]
        ax.hist(group_means, bins=60, color=color,
                edgecolor='white', alpha=0.85)
        ax.axvline(grand_mean, color=RED, linewidth=1.5, linestyle='--',
                   label=f'grand mean={grand_mean:.2f}')
        ax.set_xlabel('Group mean', fontsize=9)
        ax.set_ylabel('Number of groups', fontsize=9)
        ax.set_title(f'{label}\nGroup means  (SD={gm_std:.3f})',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: group mean vs group size (coloured by within-group SD)
        ax = axes[row_idx, 1]
        sc = ax.scatter(group_sizes, group_means,
                        c=group_stds, cmap='plasma',
                        alpha=0.4, s=12, rasterized=True)
        ax.axhline(grand_mean, color=RED, linewidth=1, linestyle='--')
        plt.colorbar(sc, ax=ax, label='Within-group SD')
        ax.set_xlabel('Group size (n observations)', fontsize=9)
        ax.set_ylabel('Group mean', fontsize=9)
        ax.set_title(f'{label}\nGroup mean vs size\n(colour = within-group SD)',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # Panel 3: Q-Q plot of group means
        ax = axes[row_idx, 2]
        (osm, osr), (slope, intercept, r) = stats.probplot(
            group_means, dist='norm', fit=True)
        ax.plot(osm, osr, '.', color=color, alpha=0.4,
                markersize=4, rasterized=True)
        ax.plot(osm, slope * np.array(osm) + intercept,
                color=RED, linewidth=1.5, label=f'r={r:.3f}')
        ax.set_xlabel('Theoretical quantiles', fontsize=9)
        ax.set_ylabel('Group mean quantiles', fontsize=9)
        ax.set_title(f'{label}\nQ-Q of group means\n'
                     f'(normal → well-identified random effect)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'H4: Distribution of group means\n'
        'Wide, normally distributed group means → strong random effect\n'
        'Tight, degenerate group means → Group Var collapses to 0',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save(fig, 'h4_group_mean_distributions.png')


def diag_residual_fe_comparison(df_t, df_s):
    """
    After-fixed-effects residual comparison.

    Fits OLS with fixed effects for each model, then plots:
      - Distribution of OLS residuals (should be similar scale if the models
        are comparable; large residuals = more unexplained noise)
      - Residual vs group mean (if residuals are systematically structured
        by group, a random intercept is justified; if not, it isn't)
    """
    print('\n' + '='*70)
    print('POST-FE RESIDUAL ANALYSIS')
    print('='*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row_idx, (label, df, fe, color) in enumerate([
        ('timing (int_y)',          df_t, 'release_speed_c + plate_x_bat_flip', BLUE),
        ('tilt (swing_path_tilt)',  df_s, 'plate_z',                            GREEN),
    ]):
        ols   = smf.ols(f'outcome ~ {fe}', data=df).fit()
        resid = ols.resid
        grp_resid_means = resid.groupby(df['_group']).mean()

        print(f'\n  {label}:')
        print(f'    OLS R²              : {ols.rsquared:.4f}')
        print(f'    Residual std        : {resid.std():.4f}')
        print(f'    SD of group resid means : {grp_resid_means.std():.4f}')
        print(f'    Kurtosis of resids  : {stats.kurtosis(resid):.3f}')

        # Panel 1: residual distribution
        ax = axes[row_idx, 0]
        ax.hist(resid.clip(lower=resid.quantile(0.01),
                           upper=resid.quantile(0.99)),
                bins=80, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel('OLS residual', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{label}\nOLS residuals  '
                     f'(std={resid.std():.3f}, R²={ols.rsquared:.3f})',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # Panel 2: distribution of per-group residual means
        # A wide distribution here means groups differ systematically in
        # their residuals → random intercept is warranted.
        # A tight spike at zero means groups are indistinguishable → collapse.
        ax = axes[row_idx, 1]
        ax.hist(grp_resid_means.clip(
                    lower=grp_resid_means.quantile(0.01),
                    upper=grp_resid_means.quantile(0.99)),
                bins=60, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(0, color=RED, linewidth=1.5, linestyle='--')
        ax.set_xlabel('Mean OLS residual per group', fontsize=9)
        ax.set_ylabel('Number of groups', fontsize=9)
        ax.set_title(
            f'{label}\nPer-group residual means\n'
            f'SD={grp_resid_means.std():.4f}  '
            f'← wide = random effect warranted',
            fontsize=10, fontweight='bold'
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        'Post-fixed-effects residual analysis\n'
        'Left: overall residual distribution  |  '
        'Right: per-group residual means\n'
        'Wide per-group residual means → systematic group structure → '
        'random intercept is warranted',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save(fig, 'residual_fe_comparison.png')


def diag_summary_table(df_t, df_s, h1_results, h2_df):
    """
    Print and save a clean summary table of all key statistics,
    and produce a single-page visual summary.
    """
    print('\n' + '='*70)
    print('SUMMARY TABLE')
    print('='*70)

    rows = []
    for label, df, fe in [
        ('timing (int_y)',         df_t, 'release_speed_c + plate_x_bat_flip'),
        ('tilt (swing_path_tilt)', df_s, 'plate_z'),
    ]:
        icc_raw, gv_raw, rv_raw, _ = null_model_icc(df)
        icc_res, gv_res, rv_res, r2 = residual_icc(df, fe)
        grp   = df.groupby('_group')['outcome']
        gm    = grp.mean()
        gs    = grp.std()
        gn    = grp.count()
        snr   = ((gm - df['outcome'].mean()).abs() / (gs + 1e-9))

        rows.append({
            'Model'                   : label,
            'n_obs'                   : len(df),
            'n_groups'                : df['_group'].nunique(),
            'median_group_size'       : int(gn.median()),
            'outcome_total_var'       : round(df['outcome'].var(), 3),
            'ICC_raw'                 : round(icc_raw, 4),
            'group_var_raw'           : round(gv_raw, 3),
            'resid_var_raw'           : round(rv_raw, 3),
            'fe_R2'                   : round(r2, 4),
            'ICC_after_FE'            : round(icc_res, 4),
            'group_var_after_FE'      : round(gv_res, 3),
            'SD_of_group_means'       : round(gm.std(), 4),
            'mean_within_group_SD'    : round(gs.mean(), 3),
            'median_SNR'              : round(snr.median(), 4),
            'pct_groups_SNR_gt1'      : round((snr > 1).mean() * 100, 1),
        })

    summary = pd.DataFrame(rows).set_index('Model').T
    print(summary.to_string())

    path = os.path.join(OUT_DIR, 'collapse_summary.csv')
    summary.to_csv(path)
    print(f'\n  → {path}')

    # ── Visual summary ────────────────────────────────────────────────────────
    metrics = [
        ('ICC_raw',            'Raw ICC\n(no FE)'),
        ('ICC_after_FE',       'Residual ICC\n(after FE)'),
        ('fe_R2',              'Fixed effects\nR²'),
        ('median_SNR',         'Median SNR\n(group signal/noise)'),
        ('pct_groups_SNR_gt1', '% groups\nSNR > 1'),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, (col, title) in zip(axes, metrics):
        vals   = [summary.loc[col, c] for c in summary.columns]
        colors = [BLUE, GREEN]
        bars   = ax.bar(summary.columns, [float(v) for v in vals],
                        color=colors, edgecolor='white', alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    float(v) + max(float(v) for v in vals) * 0.03,
                    str(v), ha='center', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticklabels(summary.columns, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle('Collapse diagnostic summary\n'
                 'Blue = timing (int_y)  |  Green = tilt (swing_path_tilt)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, 'collapse_summary_visual.png')

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Loading data...')
    df = load_and_prepare()
    df_t, df_s = build_datasets(df)

    h1_results = diag_h1_fixed_effects_absorption(df_t, df_s)
    h2_df      = diag_h2_icc_distribution(df_t, df_s)
    diag_h3_within_between_ratio(df_t, df_s)
    diag_h4_group_mean_distributions(df_t, df_s)
    diag_residual_fe_comparison(df_t, df_s)
    diag_summary_table(df_t, df_s, h1_results, h2_df)

    print(f'\nAll collapse diagnostic plots saved to {OUT_DIR}')
    print('\nINTERPRETATION GUIDE')
    print('─'*70)
    print('H1 supported → fixed effects absorb the between-group variance.')
    print('  Fix: there may be no way to recover nested structure for int_y.')
    print('  Action: use batter-only grouping for consistency.')
    print()
    print('H2 supported → ICC genuinely ≈ 0 at nested level but not at')
    print('  batter level. Action: use batter-level grouping which has ICC > 0.')
    print()
    print('H3 supported → within-group noise dominates. Median SNR << 1')
    print('  means most groups cannot be distinguished from the grand mean.')
    print('  Fix: aggregate to coarser grouping or accept OLS.')
    print()
    print('H4 supported → group mean distribution is degenerate (tight spike).')
    print('  This is the observable consequence of H1–H3; confirms the model')
    print('  correctly estimates Group Var ≈ 0 rather than making an error.')


if __name__ == '__main__':
    main()