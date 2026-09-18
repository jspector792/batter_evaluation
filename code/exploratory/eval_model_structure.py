"""
swing_variance_analysis.py
==========================
Variance explanation (R²) analysis for all four swing models.

For each outcome, fits OLS regressions for:
  - Every individual predictor
  - Every combination of predictors up to the full model
  - Pitch label (Statcast) vs pitch cluster (k-means) as sole predictor
    and as part of the full model

Also attempts the nested timing model with sparse-cell filtering,
and applies mean-centering to release_speed in every model that uses it.

Outputs
-------
  out/diagnostics/var_<outcome>_individual.png   – R² per single predictor
  out/diagnostics/var_<outcome>_combos.png       – R² heatmap over all combos
  out/diagnostics/var_pitch_label_vs_cluster.png – label vs cluster comparison
  out/diagnostics/variance_summary.csv           – full numeric table
"""

import os
import glob
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column name constants ─────────────────────────────────────────────────────
INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'

IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt'}
MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT    = IN_PLAY | FOUL
ALL_SWINGS = CONTACT | MISS

N_CLUSTERS      = 10
MIN_CELL_OBS    = 10    # minimum obs per batter/pitch_group for nested timing attempt

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE  = '#2563EB'
GREEN = '#16A34A'
RED   = '#DC2626'
GRAY  = '#6B7280'


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD + CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def load_and_cluster():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} pitches')

    df['pfx_x_arm'] = (
        df['pfx_x'] * df['p_throws'].map({'R': 1, 'L': -1}).fillna(1)
    )
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    feats = ['pfx_x_arm', 'pfx_z', 'release_speed']
    df    = df.dropna(subset=feats + ['pitch_type', 'p_throws']).copy()

    X      = df[feats].values
    scaler = StandardScaler().fit(X)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(
                scaler.transform(X))
    df['pitch_cluster'] = km.labels_.astype(str)

    # Mean-center release_speed here so it is available for all downstream use
    rs_mean = df['release_speed'].mean()
    df['release_speed_c'] = df['release_speed'] - rs_mean
    print(f'  release_speed mean-centred at {rs_mean:.2f} mph  '
          f'(column: release_speed_c)')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. R² UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _r2_ols(df, outcome, predictors):
    """
    Fit OLS and return adjusted R².
    predictors is a list of column names; categorical ones should already
    be formatted as C(col) or encoded, but here we pass raw names and let
    patsy handle them via the formula.

    Categorical predictors (non-numeric dtype or explicitly flagged) are
    wrapped in C() automatically.
    """
    terms = []
    for p in predictors:
        if df[p].dtype == object or str(df[p].dtype) == 'category':
            terms.append(f'C({p})')
        else:
            terms.append(p)

    formula = f'{outcome} ~ {" + ".join(terms)}'
    try:
        res = smf.ols(formula, data=df).fit()
        return res.rsquared, res.rsquared_adj, res.nobs
    except Exception as e:
        return np.nan, np.nan, np.nan


def _r2_all_combos(df, outcome, predictor_pool, max_combo=None):
    """
    Compute R² for every non-empty subset of predictor_pool.
    If max_combo is set, only subsets of size <= max_combo are evaluated.
    Returns a list of dicts: {predictors, n_preds, r2, r2_adj, n_obs}.
    """
    if max_combo is None:
        max_combo = len(predictor_pool)

    records = []
    for size in range(1, max_combo + 1):
        for combo in itertools.combinations(predictor_pool, size):
            combo = list(combo)
            r2, r2_adj, nobs = _r2_ols(df, outcome, combo)
            records.append({
                'predictors': ' + '.join(combo),
                'n_preds'   : size,
                'r2'        : r2,
                'r2_adj'    : r2_adj,
                'n_obs'     : nobs,
            })
    return pd.DataFrame(records).sort_values('r2', ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_individual_r2(records_df, outcome, title, fname):
    """
    Horizontal bar chart of R² for each single-predictor model,
    sorted descending.
    """
    single = (records_df[records_df['n_preds'] == 1]
              .sort_values('r2', ascending=True)
              .copy())

    fig, ax = plt.subplots(figsize=(9, max(4, len(single) * 0.45)))
    colors = [BLUE if not pd.isna(r) else GRAY for r in single['r2']]
    bars   = ax.barh(single['predictors'], single['r2'], color=colors,
                     edgecolor='white')

    for bar, val in zip(bars, single['r2']):
        if not pd.isna(val):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('R²', fontsize=11)
    ax.set_title(f'{title}\nIndividual predictor R²', fontsize=12,
                 fontweight='bold')
    ax.set_xlim(0, min(1.0, single['r2'].max() * 1.25 + 0.05))
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def _plot_combo_r2(records_df, outcome, title, fname):
    """
    Two panels:
      Left:  strip plot of R² by number of predictors (shows distribution
             of R² achievable at each model size)
      Right: top-20 combinations ranked by R², coloured by n_preds
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(records_df.head(20)) * 0.4)))

    # ── Left: R² distribution by model size ──────────────────────────────────
    ax = axes[0]
    palette = sns.color_palette('tab10', records_df['n_preds'].max())
    for size, grp in records_df.groupby('n_preds'):
        ax.scatter(grp['r2'], [size] * len(grp),
                   alpha=0.5, s=18,
                   color=palette[size - 1],
                   label=f'{size} predictor{"s" if size > 1 else ""}')
    ax.set_xlabel('R²', fontsize=11)
    ax.set_ylabel('Number of predictors', fontsize=11)
    ax.set_title('R² distribution by model complexity', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # ── Right: top-20 combinations ────────────────────────────────────────────
    ax   = axes[1]
    top  = records_df.head(20).sort_values('r2', ascending=True)
    cols = [palette[n - 1] for n in top['n_preds']]
    bars = ax.barh(range(len(top)), top['r2'], color=cols, edgecolor='white')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['predictors'], fontsize=8)
    for bar, val in zip(bars, top['r2']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.set_xlabel('R²', fontsize=11)
    ax.set_title('Top 20 predictor combinations', fontsize=11,
                 fontweight='bold')
    ax.set_xlim(0, min(1.0, top['r2'].max() * 1.15 + 0.05))
    ax.grid(axis='x', alpha=0.4)

    fig.suptitle(f'{title}  |  All predictor combinations', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def _shapley_r2(records_df, predictor_pool):
    """
    Compute the average marginal contribution of each predictor to R²,
    averaged over all subsets of the other predictors it could be added to.
    This is the Shapley value of each predictor in the cooperative game
    where the value function is R².

    For predictor x and a subset S of other predictors:
        marginal(x, S) = R²(S ∪ {x}) - R²(S)

    The Shapley value averages this over all 2^(n-1) subsets S of the
    remaining n-1 predictors, weighted equally.

    Parameters
    ----------
    records_df     : output of _r2_all_combos (must cover all subsets)
    predictor_pool : full list of predictors

    Returns
    -------
    pd.Series indexed by predictor name, values are Shapley R² contributions
    """
    # Build lookup: frozenset of predictors → R²
    r2_lookup = {}
    for _, row in records_df.iterrows():
        key = frozenset(row['predictors'].split(' + '))
        r2_lookup[key] = row['r2']
    r2_lookup[frozenset()] = 0.0   # empty model has R²=0

    n          = len(predictor_pool)
    shapley    = {p: 0.0 for p in predictor_pool}
    n_subsets  = {p: 0   for p in predictor_pool}

    for p in predictor_pool:
        others = [q for q in predictor_pool if q != p]
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                s       = frozenset(subset)
                s_with  = s | {p}
                r2_with    = r2_lookup.get(s_with, np.nan)
                r2_without = r2_lookup.get(s, np.nan)
                if not (np.isnan(r2_with) or np.isnan(r2_without)):
                    shapley[p]   += r2_with - r2_without
                    n_subsets[p] += 1

        if n_subsets[p] > 0:
            shapley[p] /= n_subsets[p]

    return pd.Series(shapley).sort_values(ascending=False)


def _plot_shapley_r2(shapley_series, title, fname):
    """
    Horizontal bar chart of average marginal R² contribution per predictor.
    """
    s = shapley_series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.5)))

    colors = [GREEN if v >= 0 else RED for v in s.values]
    bars   = ax.barh(s.index, s.values, color=colors, edgecolor='white', alpha=0.85)

    for bar, val in zip(bars, s.values):
        x_pos = val + abs(s.values).max() * 0.01 if val >= 0 \
                else val - abs(s.values).max() * 0.04
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel('Average marginal R² contribution\n'
                  '(Shapley value — averaged over all predictor subsets)',
                  fontsize=10)
    ax.set_title(f'{title}\nAverage marginal contribution to R²',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def _plot_label_vs_cluster(comparison_rows, fname):
    """
    Grouped bar chart comparing Statcast pitch_type vs k-means pitch_cluster
    as predictors across all outcomes, for both 'alone' and 'full model' cases.
    """
    df = pd.DataFrame(comparison_rows)
    outcomes  = df['outcome'].unique()
    scenarios = ['label_alone', 'cluster_alone',
                 'label_full',  'cluster_full']
    labels    = ['Statcast label\n(alone)', 'k-means cluster\n(alone)',
                 'Statcast label\n(+all vars)',  'k-means cluster\n(+all vars)']
    colors    = [BLUE, GREEN, '#1E40AF', '#15803D']

    x     = np.arange(len(outcomes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (scen, lbl, col) in enumerate(zip(scenarios, labels, colors)):
        vals = [df.loc[df['outcome'] == o, scen].values[0]
                if scen in df.columns else np.nan
                for o in outcomes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=lbl, color=col, edgecolor='white', alpha=0.88)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Statcast pitch label vs k-means cluster\nas predictors of swing outcomes',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-MODEL ANALYSES
# ══════════════════════════════════════════════════════════════════════════════

def analyse_timing(df):
    """
    Outcome: int_y  (INTERCEPT_Y)
    Predictors: release_speed_c, plate_x_bat_flip, pitch_cluster, pitch_type
    Data: all swings
    Also attempts nested grouping with sparse-cell filtering.
    """
    print('\n' + '='*70)
    print('VARIANCE ANALYSIS: TIMING (int_y)')
    print('='*70)

    needed = [INTERCEPT_Y, 'release_speed_c', 'plate_x_bat_flip',
              'pitch_cluster', 'pitch_type', 'description', 'batter']
    df_t = (df[df['description'].isin(ALL_SWINGS)]
            .dropna(subset=[c for c in needed if c in df.columns])
            .rename(columns={INTERCEPT_Y: 'int_y'})
            .copy())
    print(f'  n={len(df_t):,}')

    predictor_pool = ['release_speed_c', 'plate_x_bat_flip', 'pitch_cluster']
    records        = _r2_all_combos(df_t, 'int_y', predictor_pool)

    _plot_individual_r2(records, 'int_y',
                        'Timing model (int_y, all swings)',
                        'var_int_y_individual.png')
    _plot_combo_r2(records, 'int_y',
                   'Timing model (int_y)',
                   'var_int_y_combos.png')

    # Label vs cluster comparison (solo)
    r2_label_alone,   _, _ = _r2_ols(df_t, 'int_y', ['pitch_type'])
    r2_cluster_alone, _, _ = _r2_ols(df_t, 'int_y', ['pitch_cluster'])

    # Full model with each label type
    full_with_label   = ['release_speed_c', 'plate_x_bat_flip', 'pitch_type']
    full_with_cluster = ['release_speed_c', 'plate_x_bat_flip', 'pitch_cluster']
    r2_label_full,   _, _ = _r2_ols(df_t, 'int_y', full_with_label)
    r2_cluster_full, _, _ = _r2_ols(df_t, 'int_y', full_with_cluster)

    print(f'  pitch_type  alone R²={r2_label_alone:.4f}  '
          f'full R²={r2_label_full:.4f}')
    print(f'  pitch_cluster alone R²={r2_cluster_alone:.4f}  '
          f'full R²={r2_cluster_full:.4f}')

    # ── Nested timing model attempt with sparse-cell filtering ────────────────
    print(f'\n  Attempting nested timing model '
          f'(min {MIN_CELL_OBS} obs per batter×cluster cell)...')
    df_t['_group'] = (df_t['batter'].astype(str) + '__'
                      + df_t['pitch_cluster'].astype(str))
    cell_counts    = df_t.groupby('_group').size()
    valid_groups   = cell_counts[cell_counts >= MIN_CELL_OBS].index
    df_nested      = df_t[df_t['_group'].isin(valid_groups)].copy()
    pct_kept       = len(df_nested) / len(df_t) * 100
    print(f'    Rows after filtering: {len(df_nested):,} '
          f'({pct_kept:.1f}% of all-swing data)')
    print(f'    Groups remaining:     {df_nested["_group"].nunique():,}')

    nested_success = False
    if len(df_nested) > 1000 and df_nested['_group'].nunique() > 10:
        try:
            model  = smf.mixedlm(
                'int_y ~ release_speed_c + plate_x_bat_flip + C(pitch_cluster)',
                data=df_nested,
                groups=df_nested['_group'],
            )
            result = model.fit(reml=True, method='lbfgs')
            gv     = float(result.cov_re.iloc[0, 0])
            print(f'    Nested model converged. Group Var = {gv:.4f}')
            nested_success = True
        except Exception as e:
            print(f'    Nested model failed: {e}')
    else:
        print('    Insufficient data after filtering — skipping.')

    if not nested_success:
        print('    → Nested timing model remains infeasible with this dataset.')
        print('      Recommendation: use batter-only grouping for consistency.')

    return {
        'outcome'       : 'int_y',
        'label_alone'   : r2_label_alone,
        'cluster_alone' : r2_cluster_alone,
        'label_full'    : r2_label_full,
        'cluster_full'  : r2_cluster_full,
        'records'       : records,
    }


def analyse_swing_tilt(df):
    """
    Outcome: swing_path_tilt
    Predictors: plate_z, release_speed_c, pitch_cluster, pitch_type
    Data: misses only
    """
    print('\n' + '='*70)
    print('VARIANCE ANALYSIS: SWING PATH TILT (misses)')
    print('='*70)

    needed = ['swing_path_tilt', 'plate_z', 'release_speed_c',
              'pitch_cluster', 'pitch_type', 'description']
    df_s = (df[df['description'].isin(MISS)]
            .dropna(subset=[c for c in needed if c in df.columns])
            .copy())
    print(f'  n={len(df_s):,}')

    predictor_pool = ['plate_z', 'release_speed_c', 'pitch_cluster']
    records        = _r2_all_combos(df_s, 'swing_path_tilt', predictor_pool)

    _plot_individual_r2(records, 'swing_path_tilt',
                        'Swing tilt model (misses)',
                        'var_swing_tilt_individual.png')
    _plot_combo_r2(records, 'swing_path_tilt',
                   'Swing tilt model (misses)',
                   'var_swing_tilt_combos.png')

    r2_label_alone,   _, _ = _r2_ols(df_s, 'swing_path_tilt', ['pitch_type'])
    r2_cluster_alone, _, _ = _r2_ols(df_s, 'swing_path_tilt', ['pitch_cluster'])
    r2_label_full,   _, _  = _r2_ols(df_s, 'swing_path_tilt',
                                      ['plate_z', 'release_speed_c', 'pitch_type'])
    r2_cluster_full, _, _  = _r2_ols(df_s, 'swing_path_tilt',
                                      ['plate_z', 'release_speed_c', 'pitch_cluster'])

    print(f'  pitch_type    alone R²={r2_label_alone:.4f}  '
          f'full R²={r2_label_full:.4f}')
    print(f'  pitch_cluster alone R²={r2_cluster_alone:.4f}  '
          f'full R²={r2_cluster_full:.4f}')

    return {
        'outcome'       : 'swing_path_tilt',
        'label_alone'   : r2_label_alone,
        'cluster_alone' : r2_cluster_alone,
        'label_full'    : r2_label_full,
        'cluster_full'  : r2_cluster_full,
        'records'       : records,
    }


def analyse_exit_velo(df):
    """
    Outcome: exit_velo (launch_speed)
    Predictors: intercept_y, intercept_x, attack_angle,
                swing_path_tilt, plate_z, release_speed_c,
                bat_speed (if available), pitch_cluster, pitch_type
    Data: in-play only
    """
    print('\n' + '='*70)
    print('VARIANCE ANALYSIS: EXIT VELOCITY (contact)')
    print('='*70)

    df_c = (df[df['description'].isin(IN_PLAY)]
            .rename(columns={
                INTERCEPT_X   : 'intercept_x',
                INTERCEPT_Y   : 'intercept_y',
                'launch_speed': 'exit_velo',
            })
            .dropna(subset=['exit_velo', 'intercept_x', 'intercept_y',
                            'attack_angle', 'plate_z', 'release_speed_c',
                            'pitch_cluster', 'pitch_type'])
            .copy())
    print(f'  n={len(df_c):,}')

    has_spt = df_c['swing_path_tilt'].notna().sum() > 100
    has_bs  = ('bat_speed' in df_c.columns and
               df_c['bat_speed'].notna().sum() > 100)

    base_pool = ['intercept_y', 'intercept_x', 'attack_angle',
                 'plate_z', 'release_speed_c', 'pitch_cluster']
    if has_spt:
        df_c      = df_c.dropna(subset=['swing_path_tilt'])
        base_pool = base_pool + ['swing_path_tilt']
    if has_bs:
        df_c      = df_c.dropna(subset=['bat_speed'])
        base_pool = base_pool + ['bat_speed']
        print(f'  bat_speed available: n={len(df_c):,} after filtering')
    else:
        print('  bat_speed not available — skipping')
    pool = base_pool

    records = _r2_all_combos(df_c, 'exit_velo', pool)
    shapley = _shapley_r2(records, pool)

    _plot_individual_r2(records, 'exit_velo',
                        'Exit velocity model (contact)',
                        'var_exit_velo_individual.png')
    _plot_combo_r2(records, 'exit_velo',
                   'Exit velocity model (contact)',
                   'var_exit_velo_combos.png')
    _plot_shapley_r2(shapley,
                     'Exit velocity model (contact)',
                     'var_exit_velo_shapley.png')

    print('  Shapley R² contributions:')
    for pred, val in shapley.items():
        print(f'    {pred:25s}: {val:.4f}')

    r2_label_alone,   _, _ = _r2_ols(df_c, 'exit_velo', ['pitch_type'])
    r2_cluster_alone, _, _ = _r2_ols(df_c, 'exit_velo', ['pitch_cluster'])

    full_vars = [v for v in pool if v != 'pitch_cluster']
    r2_label_full,   _, _ = _r2_ols(df_c, 'exit_velo', full_vars + ['pitch_type'])
    r2_cluster_full, _, _ = _r2_ols(df_c, 'exit_velo', full_vars + ['pitch_cluster'])

    print(f'  pitch_type    alone R²={r2_label_alone:.4f}  '
          f'full R²={r2_label_full:.4f}')
    print(f'  pitch_cluster alone R²={r2_cluster_alone:.4f}  '
          f'full R²={r2_cluster_full:.4f}')

    return {
        'outcome'       : 'exit_velo',
        'label_alone'   : r2_label_alone,
        'cluster_alone' : r2_cluster_alone,
        'label_full'    : r2_label_full,
        'cluster_full'  : r2_cluster_full,
        'records'       : records,
        'shapley'       : shapley,
    }


def analyse_launch_angle(df):
    """
    Outcome: launch_angle
    Predictors: intercept_y, attack_angle, swing_path_tilt,
                plate_z, release_speed_c, bat_speed (if available),
                pitch_cluster, pitch_type
    Data: in-play only
    """
    print('\n' + '='*70)
    print('VARIANCE ANALYSIS: LAUNCH ANGLE (contact)')
    print('='*70)

    df_c = (df[df['description'].isin(IN_PLAY)]
            .rename(columns={
                INTERCEPT_Y: 'intercept_y',
            })
            .dropna(subset=['launch_angle', 'intercept_y', 'attack_angle',
                            'plate_z', 'release_speed_c',
                            'pitch_cluster', 'pitch_type'])
            .copy())
    print(f'  n={len(df_c):,}')

    has_spt = df_c['swing_path_tilt'].notna().sum() > 100
    has_bs  = ('bat_speed' in df_c.columns and
               df_c['bat_speed'].notna().sum() > 100)

    base_pool = ['intercept_y', 'attack_angle',
                 'plate_z', 'release_speed_c', 'pitch_cluster']
    if has_spt:
        df_c      = df_c.dropna(subset=['swing_path_tilt'])
        base_pool = base_pool + ['swing_path_tilt']
    if has_bs:
        df_c      = df_c.dropna(subset=['bat_speed'])
        base_pool = base_pool + ['bat_speed']
        print(f'  bat_speed available: n={len(df_c):,} after filtering')
    else:
        print('  bat_speed not available — skipping')
    pool = base_pool

    records = _r2_all_combos(df_c, 'launch_angle', pool)
    shapley = _shapley_r2(records, pool)

    _plot_individual_r2(records, 'launch_angle',
                        'Launch angle model (contact)',
                        'var_launch_angle_individual.png')
    _plot_combo_r2(records, 'launch_angle',
                   'Launch angle model (contact)',
                   'var_launch_angle_combos.png')
    _plot_shapley_r2(shapley,
                     'Launch angle model (contact)',
                     'var_launch_angle_shapley.png')

    print('  Shapley R² contributions:')
    for pred, val in shapley.items():
        print(f'    {pred:25s}: {val:.4f}')

    r2_label_alone,   _, _ = _r2_ols(df_c, 'launch_angle', ['pitch_type'])
    r2_cluster_alone, _, _ = _r2_ols(df_c, 'launch_angle', ['pitch_cluster'])

    full_vars = [v for v in pool if v != 'pitch_cluster']
    r2_label_full,   _, _ = _r2_ols(df_c, 'launch_angle',
                                     full_vars + ['pitch_type'])
    r2_cluster_full, _, _ = _r2_ols(df_c, 'launch_angle',
                                     full_vars + ['pitch_cluster'])

    print(f'  pitch_type    alone R²={r2_label_alone:.4f}  '
          f'full R²={r2_label_full:.4f}')
    print(f'  pitch_cluster alone R²={r2_cluster_alone:.4f}  '
          f'full R²={r2_cluster_full:.4f}')

    return {
        'outcome'       : 'launch_angle',
        'label_alone'   : r2_label_alone,
        'cluster_alone' : r2_cluster_alone,
        'label_full'    : r2_label_full,
        'cluster_full'  : r2_cluster_full,
        'records'       : records,
        'shapley'       : shapley,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(analyses):
    """Write a CSV of full-model and individual R² values."""
    rows = []
    for a in analyses:
        outcome  = a['outcome']
        records  = a['records']
        # Individual predictors
        for _, row in records[records['n_preds'] == 1].iterrows():
            rows.append({
                'outcome'   : outcome,
                'model_type': 'individual',
                'predictors': row['predictors'],
                'n_preds'   : 1,
                'r2'        : row['r2'],
                'r2_adj'    : row['r2_adj'],
            })
        # Full model (all predictors)
        full = records[records['n_preds'] == records['n_preds'].max()].iloc[0]
        rows.append({
            'outcome'   : outcome,
            'model_type': 'full',
            'predictors': full['predictors'],
            'n_preds'   : full['n_preds'],
            'r2'        : full['r2'],
            'r2_adj'    : full['r2_adj'],
        })
        # Label vs cluster
        for scenario in ['label_alone', 'cluster_alone',
                         'label_full',  'cluster_full']:
            rows.append({
                'outcome'   : outcome,
                'model_type': scenario,
                'predictors': 'pitch_type' if 'label' in scenario
                              else 'pitch_cluster',
                'n_preds'   : np.nan,
                'r2'        : a[scenario],
                'r2_adj'    : np.nan,
            })

    out_df = pd.DataFrame(rows)
    path   = os.path.join(OUT_DIR, 'variance_summary.csv')
    out_df.to_csv(path, index=False)
    print(f'\n  → {path}')
    return out_df


def plot_summary_overview(analyses):
    """
    One-page overview: full-model R² across all four outcomes,
    with individual-predictor R² stacked to show relative contribution.
    """
    outcomes = [a['outcome'] for a in analyses]
    fig, axes = plt.subplots(1, len(analyses), figsize=(5 * len(analyses), 6),
                              sharey=False)

    for ax, a in zip(axes, analyses):
        indiv  = (a['records'][a['records']['n_preds'] == 1]
                  .set_index('predictors')['r2']
                  .sort_values(ascending=False))
        full_r2 = a['records'].iloc[0]['r2']   # highest R² (full model)

        colors = sns.color_palette('tab10', len(indiv))
        bars   = ax.bar(range(len(indiv)), indiv.values,
                        color=colors, edgecolor='white', alpha=0.85)
        ax.axhline(full_r2, color=RED, linewidth=1.8, linestyle='--',
                   label=f'Full model R²={full_r2:.3f}')

        ax.set_xticks(range(len(indiv)))
        ax.set_xticklabels(indiv.index, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('R²', fontsize=10)
        ax.set_title(a['outcome'], fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.35)

        for bar, v in zip(bars, indiv.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.002, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    fig.suptitle('Variance explained (R²) — individual predictors vs full model',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'var_overview.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_and_cluster()

    a_timing = analyse_timing(df)
    a_tilt   = analyse_swing_tilt(df)
    a_ev     = analyse_exit_velo(df)
    a_la     = analyse_launch_angle(df)

    analyses = [a_timing, a_tilt, a_ev, a_la]

    # Label vs cluster comparison plot
    _plot_label_vs_cluster(
        [{'outcome'       : a['outcome'],
          'label_alone'   : a['label_alone'],
          'cluster_alone' : a['cluster_alone'],
          'label_full'    : a['label_full'],
          'cluster_full'  : a['cluster_full']}
         for a in analyses],
        fname='var_pitch_label_vs_cluster.png',
    )

    save_summary(analyses)
    plot_summary_overview(analyses)

    print(f'\nAll variance analysis outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()