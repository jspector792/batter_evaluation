"""
run_frozen_models.py
====================
Runs the two frozen MERF models and writes all outputs:

  1. int_y MERF   — timing model with platoon effect
  2. barrel_distance MERF (v2, 20°-centered) — barrel placement model

For each model:
  - Fits MERF and saves the fitted object to disk (joblib) so it can be
    reloaded without refitting
  - Writes per-batter scores and random intercepts to a single combined
    scoring CSV
  - Writes predictions back into the source parquet files so that every
    swing row has int_y_predicted and barrel_distance_predicted alongside
    the raw data

Saved model files
-----------------
  {OUT_DIR}/int_y_merf_model.joblib
  {OUT_DIR}/barrel_merf_model.joblib

  To reload a saved model without refitting:
    import joblib
    mrf = joblib.load('path/to/int_y_merf_model.joblib')
    fitted = mrf.predict(X, Z, clusters)

Combined scoring file
---------------------
  {OUT_DIR}/batter_scores_combined.csv

  Columns:
    batter
    timing_random_intercept   — b_i from int_y model (negative = earlier contact)
    timing_score              — 1 / mean(|residual|) from int_y model
    timing_n_swings           — swing count used for timing score
    barrel_random_intercept   — b_i from barrel model (negative = better placement)
    barrel_placement_score    — 1 / mean(|residual|) from barrel model
    barrel_n_events           — event count used for barrel score
    barrel_n_miss             — miss events contributing to barrel score
    barrel_n_contact          — contact events contributing to barrel score

Prediction columns written back to parquet
------------------------------------------
  int_y_predicted             — MERF fitted value on all swings (NaN on non-swings)
  barrel_distance_v2          — computed barrel_distance outcome (NaN on non-swings)
  barrel_distance_predicted   — MERF fitted value on all swings (NaN on non-swings)

Physical constants
------------------
  C = (bat_diameter/2) + (ball_diameter/2) = 1.305 + 1.450 = 2.755 inches
  Barrel zone: launch angle in [8°, 32°]

Usage
-----
  python run_frozen_models.py [--skip-fit] [--skip-parquet]

  --skip-fit      Load saved model files instead of refitting (saves time on reruns)
  --skip-parquet  Skip writing predictions back to parquet files

Requires: merf, xgboost, joblib, scikit-learn, pandas, numpy
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from merf import MERF

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS
CONTACT    = IN_PLAY | FOUL

# ── Physical constants (barrel_distance v2) ────────────────────────────────────
BAT_DIAMETER  = 2.61
BALL_DIAMETER = 2.90
C      = (BAT_DIAMETER / 2) + (BALL_DIAMETER / 2)   # 2.755 inches
LA_CENTER = 20.0   # degrees — zero-error point for v2 formula

# ── Scoring ────────────────────────────────────────────────────────────────────
MIN_PA = 400   # minimum AB proxy for a batter to appear in scoring output

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# Each config dict fully specifies one model so the shared fitting pipeline
# can handle both without branching logic in main().
# ══════════════════════════════════════════════════════════════════════════════

INT_Y_CONFIG = dict(
    name        = 'int_y',
    outcome_col = 'int_y',
    cluster_col = 'batter',
    features    = [
        'release_speed_c',
        'plate_x_bat_flip',
        'plate_z',
        'pfx_x_bat_flip',
        'pfx_z',
        'same_hand',
    ],
    batter_agg_features = ['bat_speed', 'release_speed'],
    xgb_params  = dict(
        n_estimators     = 400,
        max_depth        = 5,
        learning_rate    = 0.04,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    ),
    max_iter         = 15,
    model_path       = os.path.join(OUT_DIR, 'int_y_merf_model.joblib'),
    pred_col         = 'int_y_predicted',
    score_prefix     = 'timing',
    score_col        = 'timing_score',
    re_col           = 'timing_random_intercept',
    n_col            = 'timing_n_swings',
)

BARREL_CONFIG = dict(
    name        = 'barrel',
    outcome_col = 'barrel_distance_v2',
    cluster_col = 'batter',
    features    = [
        'release_speed_c',
        'plate_x_bat_flip',
        'plate_z',
        'intercept_y',
    ],
    batter_agg_features = [],
    xgb_params  = dict(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    ),
    max_iter         = 20,
    model_path       = os.path.join(OUT_DIR, 'barrel_merf_model.joblib'),
    pred_col         = 'barrel_distance_predicted',
    score_prefix     = 'barrel',
    score_col        = 'barrel_placement_score',
    re_col           = 'barrel_random_intercept',
    n_col            = 'barrel_n_events',
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SHARED DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def compute_barrel_distance_v2(df: pd.DataFrame) -> pd.Series:
    bd = pd.Series(np.nan, index=df.index, dtype=float)

    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)

    # Misses: raw miss_distance (in inches) shifted up by C
    # Verify miss_distance is in expected range before applying
    valid_miss = is_miss & df['miss_distance'].notna() & (df['miss_distance'] >= 0)
    bd.loc[valid_miss] = df.loc[valid_miss, 'miss_distance'] + C

    # Contact: continuous penalty from 20°, NaN if launch_angle missing
    valid_contact = is_contact & df['launch_angle'].notna()
    la_rad = np.deg2rad(df.loc[valid_contact, 'launch_angle'])
    bd.loc[valid_contact] = np.abs(C * np.sin(la_rad - np.deg2rad(LA_CENTER)))

    # Sanity check: on misses, barrel_distance_v2 must be >= C
    # (since miss_distance >= 0, so miss_distance + C >= C = 2.755)
    n_bad = (bd[valid_miss] < C).sum()
    if n_bad > 0:
        print(f'  WARNING: {n_bad} miss rows have barrel_distance_v2 < C '
              f'({C:.3f}) — check miss_distance column')

    return bd


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load all parquet files, compute all derived columns needed by both models,
    and return:
      df       — full dataset with original index preserved (for parquet writeback)
      files    — list of parquet file paths (for writeback)
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    frames = []
    for f in files:
        part = pd.read_parquet(f)
        part['_source_file'] = f          # track which file each row came from
        frames.append(part)
    df = pd.concat(frames, ignore_index=False)   # preserve original index
    print(f'Loaded {len(df):,} total pitches from {len(files)} file(s)')

    # ── String types ──────────────────────────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── plate_x_bat_flip ──────────────────────────────────────────────────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    # ── pfx_x_bat_flip ────────────────────────────────────────────────────────
    if 'pfx_x_bat_flip' not in df.columns:
        df['pfx_x_bat_flip'] = (
            df['pfx_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    # ── abs_pfx_x ─────────────────────────────────────────────────────────────
    df['abs_pfx_x'] = df['pfx_x'].abs()

    # ── release_speed_c ───────────────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── Rename intercept_y ────────────────────────────────────────────────────
    if INTERCEPT_Y_COL in df.columns:
        df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── Platoon ───────────────────────────────────────────────────────────────
    if 'p_throws' in df.columns and 'stand' in df.columns:
        df['same_hand'] = (
            ((df['stand'] == 'R') & (df['p_throws'] == 'R')) |
            ((df['stand'] == 'L') & (df['p_throws'] == 'L'))
        ).astype(float)
        print(f'same_hand rate: {df["same_hand"].mean():.3f}')
    else:
        print('WARNING: p_throws or stand missing — same_hand set to 0')
        df['same_hand'] = 0.0

    # ── int_y (rename) ────────────────────────────────────────────────────────
    df = df.rename(columns={INTERCEPT_Y_COL: 'int_y'}, errors='ignore')
    if 'int_y' not in df.columns and 'intercept_y' in df.columns:
        # intercept_y was already renamed above; int_y is the timing outcome
        # Both models need intercept_y as a feature and int_y as an outcome —
        # they are the SAME column, just used in different roles.
        # The barrel model uses it as a predictor; the timing model as the outcome.
        df['int_y'] = df['intercept_y']

    # ── barrel_distance_v2 ────────────────────────────────────────────────────
    df['barrel_distance_v2'] = compute_barrel_distance_v2(df)
    print(f'barrel_distance_v2: '
          f'non-null={df["barrel_distance_v2"].notna().sum():,}  '
          f'mean={df["barrel_distance_v2"].mean():.3f}')

    # ── AB proxy (for scoring filter) ─────────────────────────────────────────
    bip_mask = df['description'].isin(IN_PLAY)
    k_mask   = df['events'] == 'strikeout'
    bip_ct   = df[bip_mask].groupby('batter').size().rename('n_bip')
    k_ct     = df[k_mask].groupby('batter').size().rename('n_k')
    ab_proxy = (pd.concat([bip_ct, k_ct], axis=1)
                  .fillna(0).astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])
                  [['ab_proxy']].reset_index())
    df = df.merge(ab_proxy, on='batter', how='left')
    df['ab_proxy'] = df['ab_proxy'].fillna(0)

    return df, files


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MODELLING SUBSET FOR ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_subset(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    From the full dataset, build the modelling subset and MERF inputs for
    one model config. Returns (X, Z, clusters, y, X_cols, sub).

    sub retains the original df index so predictions can be joined back.
    """
    outcome_col = cfg['outcome_col']
    cluster_col = cfg['cluster_col']
    features    = cfg['features']
    agg_feats   = cfg['batter_agg_features']

    # Filter to relevant events
    if outcome_col == 'int_y':
        event_mask = df['description'].isin(ALL_SWINGS)
    else:
        # barrel_distance_v2 is NaN on non-swings and contact with missing LA
        event_mask = df['description'].isin(ALL_SWINGS)

    sub = df[event_mask].copy()

    # Batter aggregate features
    if agg_feats:
        available = [f for f in agg_feats if f in sub.columns]
        if available:
            agg = (sub.groupby('batter')[available]
                      .mean().add_prefix('batter_mean_').reset_index())
            sub = sub.merge(agg, on='batter', how='left')

    # One-hot pitch_type
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)

    # Drop rows missing outcome or any feature
    needed = [outcome_col, cluster_col] + features
    n_before = len(sub)
    sub = sub.dropna(subset=needed).copy()
    print(f'  [{cfg["name"]}] modelling subset: {len(sub):,} rows '
          f'(dropped {n_before - len(sub):,} for NA)')

    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols = features + pt_dummies
    if agg_feats:
        X_cols += [f'batter_mean_{f}' for f in agg_feats
                   if f'batter_mean_{f}' in sub.columns]

    X        = sub[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub), 1))
    clusters = sub[cluster_col].reset_index(drop=True)
    y        = sub[outcome_col].reset_index(drop=True)

    return X, Z, clusters, y, X_cols, sub


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT OR LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_or_load(X, Z, clusters, y, cfg: dict, skip_fit: bool) -> MERF:
    """
    If skip_fit=True and the model file exists, load and return it.
    Otherwise fit MERF, save to disk with joblib, and return it.

    joblib is preferred over pickle for objects containing large numpy arrays
    (like XGBoost trees) because it uses memory-mapped files and compresses
    efficiently. The saved file can be reloaded with:
        import joblib
        mrf = joblib.load(cfg['model_path'])
    """
    model_path = cfg['model_path']

    if skip_fit and os.path.exists(model_path):
        print(f'  [{cfg["name"]}] Loading saved model from {model_path}')
        return joblib.load(model_path)

    print(f'  [{cfg["name"]}] Fitting MERF '
          f'(n={len(y):,}, features={X.shape[1]}, max_iter={cfg["max_iter"]})')
    print('  GLL should increase and plateau at convergence.\n')

    xgb = XGBRegressor(**cfg['xgb_params'])
    mrf = MERF(fixed_effects_model=xgb, max_iterations=cfg['max_iter'])
    mrf.fit(X, Z, clusters, y)

    joblib.dump(mrf, model_path, compress=3)
    print(f'  [{cfg["name"]}] Model saved → {model_path}')

    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list, cfg: dict):
    resid = y.values - fitted
    name  = cfg['name']
    color = BLUE

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{name} – MERF (XGBoost fixed effects)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.08, s=3, color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.scatter(y, fitted, alpha=0.08, s=3, color=color, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y, fitted)[0, 1]
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    gll = mrf.gll_history
    ax.plot(range(1, len(gll) + 1), gll,
            color=GREEN, linewidth=2, marker='o', markersize=5)
    ax.set_xlabel('EM Iteration'); ax.set_ylabel('GLL')
    ax.set_title('MERF convergence (GLL per EM iteration)', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lo, hi = np.percentile(resid, 0.5), np.percentile(resid, 99.5)
    ax.hist(np.clip(resid, lo, hi), bins=80,
            color=color, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (σ = {resid.std():.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{name}_merf_diagnostics.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    imp = pd.Series(mrf.trained_fe_model.feature_importances_,
                    index=X_cols).sort_values(ascending=True)
    n = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=GREEN, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'{name} – XGBoost feature importance (top 25)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{name}_merf_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      scores_df   — per-batter score (1/mean|resid|), n_events, ab_proxy
      re_df       — per-batter random intercept from mrf.trained_b
    Both filtered to batters with ab_proxy >= MIN_PA.
    """
    outcome_col = cfg['outcome_col']
    cluster_col = cfg['cluster_col']

    ab_proxy_map = sub.groupby(cluster_col)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub[outcome_col].values - fitted)

    agg_dict = dict(
        n_events       = ('abs_resid', 'size'),
        mean_abs_resid = ('abs_resid', 'mean'),
        ab_proxy       = ('ab_proxy', 'first'),
    )

    base = pd.DataFrame({
        'batter':    clusters.values,
        'abs_resid': resid_abs,
        'ab_proxy':  sub['ab_proxy'].values,
    }).groupby('batter').agg(**agg_dict)

    # For barrel model, add miss/contact breakdown
    if cfg['name'] == 'barrel':
        event_counts = pd.DataFrame({
            'batter':     clusters.values,
            'is_miss':    sub['description'].isin(MISS).values,
            'is_contact': sub['description'].isin(CONTACT).values,
        }).groupby('batter').agg(
            n_miss    = ('is_miss',    'sum'),
            n_contact = ('is_contact', 'sum'),
        )
        base = base.join(event_counts)

    scores_df = (base
                 .loc[lambda d: d.index.isin(qualifying)]
                 .assign(**{cfg['score_col']: lambda d: 1.0 / d['mean_abs_resid']})
                 .rename(columns={'n_events': cfg['n_col']})
                 .sort_values(cfg['score_col'], ascending=False)
                 .reset_index())

    print(f'\n  [{cfg["name"]}] Scores for {len(scores_df):,} batters '
          f'(≥{MIN_PA} AB proxy):')
    show_cols = ['batter', cfg['score_col'], 'mean_abs_resid', cfg['n_col']]
    print(scores_df[show_cols].head(10).to_string(index=False))

    return scores_df


def build_combined_scores(timing_scores: pd.DataFrame,
                          timing_re: pd.DataFrame,
                          barrel_scores: pd.DataFrame,
                          barrel_re: pd.DataFrame) -> pd.DataFrame:
    """
    Merge timing and barrel scores + random intercepts into a single wide CSV.
    """
    # Rename score columns
    t = (timing_scores[['batter', 'timing_score', 'timing_n_swings',
                         'mean_abs_resid']]
         .rename(columns={'mean_abs_resid': 'timing_mean_abs_resid'}))

    b_cols = ['batter', 'barrel_placement_score', 'barrel_n_events',
              'mean_abs_resid']
    if 'n_miss' in barrel_scores.columns:
        b_cols += ['n_miss', 'n_contact']
    b = (barrel_scores[b_cols]
         .rename(columns={
             'mean_abs_resid': 'barrel_mean_abs_resid',
             'n_miss':         'barrel_n_miss',
             'n_contact':      'barrel_n_contact',
         }))

    t_re = timing_re.rename(columns={'random_intercept': 'timing_random_intercept'})
    b_re = barrel_re.rename(columns={'random_intercept': 'barrel_random_intercept'})

    combined = (t
                .merge(t_re[['batter', 'timing_random_intercept']],
                       on='batter', how='outer')
                .merge(b, on='batter', how='outer')
                .merge(b_re[['batter', 'barrel_random_intercept']],
                       on='batter', how='outer'))

    # Canonical column order
    ordered = [
        'batter',
        'timing_random_intercept', 'timing_score',
        'timing_n_swings',         'timing_mean_abs_resid',
        'barrel_random_intercept', 'barrel_placement_score',
        'barrel_n_events',         'barrel_mean_abs_resid',
    ]
    if 'barrel_n_miss' in combined.columns:
        ordered += ['barrel_n_miss', 'barrel_n_contact']

    combined = combined[[c for c in ordered if c in combined.columns]]

    path = os.path.join(OUT_DIR, 'batter_scores_combined.csv')
    combined.to_csv(path, index=False)
    print(f'\nCombined scores → {path}  ({len(combined):,} batters)')
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 6. WRITE PREDICTIONS BACK TO PARQUET
# ══════════════════════════════════════════════════════════════════════════════

def write_predictions_to_parquet(df: pd.DataFrame, files: list[str]):
    pred_cols = ['int_y_predicted', 'barrel_distance_v2',
                 'barrel_distance_predicted']

    print(f'\nWriting predictions back to {len(files)} parquet file(s)...')

    # df._source_file tracks which rows belong to which file
    for fpath in files:
        part = pd.read_parquet(fpath)

        # Get the rows from df that came from this file, in original order
        mask = df['_source_file'] == fpath
        df_part = df[mask].reset_index(drop=True)  # align to 0-based for part

        for col in pred_cols:
            if col in df_part.columns:
                part[col] = df_part[col].values   # .values avoids index alignment issues

        part.to_parquet(fpath, index=False)
        n_int_y  = part['int_y_predicted'].notna().sum()
        n_barrel = part['barrel_distance_predicted'].notna().sum()
        print(f'  {os.path.basename(fpath)}: '
              f'int_y_predicted={n_int_y:,}  '
              f'barrel_distance_predicted={n_barrel:,}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Run frozen MERF models')
    parser.add_argument('--skip-fit',     action='store_true',
                        help='Load saved models instead of refitting')
    parser.add_argument('--skip-parquet', action='store_true',
                        help='Skip writing predictions back to parquet files')
    args = parser.parse_args()

    print('='*70)
    print('Frozen MERF Models — int_y (timing) + barrel_distance (placement)')
    print('='*70)

    # ── Load and prepare data ──────────────────────────────────────────────────
    df, files = prepare_data()

    # ── Sanity-write parquet files before model fitting ───────────────────────────
    # Write barrel_distance_v2 (and empty prediction columns) back to parquet now
    # so you can inspect the barrel distance calculation while models are fitting.
    # Prediction columns will be NaN at this stage — they get filled after fitting.

    df['int_y_predicted']           = np.nan
    df['barrel_distance_predicted'] = np.nan

    print('\nWriting barrel_distance_v2 to parquet (pre-fit sanity check)...')
    for fpath in files:
        part = pd.read_parquet(fpath)
        mask     = df['_source_file'] == fpath
        df_part  = df[mask].reset_index(drop=True)
        for col in ['barrel_distance_v2', 'int_y_predicted', 'barrel_distance_predicted']:
            if col in df_part.columns:
                part[col] = df_part[col].values
        part.to_parquet(fpath, index=False)
        n_bd = part['barrel_distance_v2'].notna().sum()
        print(f'  {os.path.basename(fpath)}: barrel_distance_v2 non-null={n_bd:,}')

    print('Pre-fit parquet write complete. Check barrel_distance_v2 now.\n')

    all_re = {}

    for cfg in [INT_Y_CONFIG, BARREL_CONFIG]:
        print(f'\n{"="*70}')
        print(f'MODEL: {cfg["name"]}')
        print('='*70)

        # Build modelling subset
        X, Z, clusters, y, X_cols, sub = build_subset(df, cfg)

        # Fit or load
        mrf = fit_or_load(X, Z, clusters, y, cfg, skip_fit=args.skip_fit)

        # Predict
        fitted = mrf.predict(X, Z, clusters)

        # R²
        ss_res = np.sum((y.values - fitted) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f'\n  R²: {r2:.4f}')

        # Write predictions back into df on the original index
        fitted_series = pd.Series(fitted, index=sub.index)
        df.loc[sub.index, cfg['pred_col']] = fitted_series

        # Also write barrel_distance_v2 (the outcome) if this is the barrel model
        if cfg['name'] == 'barrel':
            # barrel_distance_v2 is already on df from prepare_data
            pass

        # Random intercepts
        re_df = (mrf.trained_b
                   .rename(columns={0: 'random_intercept'})
                   .reset_index()
                   .rename(columns={'index': cfg['cluster_col']}))
        all_re[cfg['name']] = re_df

        re_path = os.path.join(OUT_DIR, f'{cfg["name"]}_merf_re.csv')
        re_df.to_csv(re_path, index=False)
        print(f'\n  Random intercepts → {re_path}')

        # Scores
        scores_df = compute_scores(sub, fitted, clusters, cfg)
        all_re[f'{cfg["name"]}_scores'] = scores_df

        # Diagnostics
        diagnostic_plots(y, fitted, mrf, X_cols, cfg)

    # ── Combined scoring file ──────────────────────────────────────────────────
    combined = build_combined_scores(
        timing_scores = all_re['int_y_scores'],
        timing_re     = all_re['int_y'],
        barrel_scores = all_re['barrel_scores'],
        barrel_re     = all_re['barrel'],
    )

    # ── Write predictions to parquet ───────────────────────────────────────────
    if not args.skip_parquet:
        write_predictions_to_parquet(df, files)
    else:
        print('\n--skip-parquet set: skipping parquet writeback')

    print(f'\nAll outputs in {OUT_DIR}')
    print('Key files:')
    for fname in [
        'int_y_merf_model.joblib',
        'barrel_merf_model.joblib',
        'batter_scores_combined.csv',
        'int_y_merf_diagnostics.png',
        'barrel_merf_diagnostics.png',
    ]:
        p = os.path.join(OUT_DIR, fname)
        status = '✓' if os.path.exists(p) else '⏳ (will exist after run)'
        print(f'  {status}  {p}')


if __name__ == '__main__':
    main()