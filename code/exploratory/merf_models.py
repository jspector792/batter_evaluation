"""
barrel_distance_merf.py
=======================
Mixed Effects Random Forest (MERF) model predicting barrel_distance on all
swing events (swinging strikes, fouls, and balls in play), using XGBoost as
the fixed-effects learner.

barrel_distance is a unified barrel-placement metric computed by
add_barrel_distance.py. It puts contact and miss events on the same scale:

  swing and miss  → miss_distance + C          (C = sum of radii, 2.755 in)
  contact, 8-32°  → 0                          (barrel zone, no penalty)
  contact, la<8°  → |C * sin(la - 8°)|         (below zone, grounder penalty)
  contact, la>32° → |C * sin(la - 32°)|        (above zone, popup penalty)

Why MERF instead of plain XGBoost?
-----------------------------------
Standard XGBoost treats every observation as i.i.d. A batter appears hundreds
of times in the data, so observations are clustered — ignoring that inflates
apparent fit and conflates pitch-difficulty effects with batter skill. MERF
separates the two:

  y_ij = f(X_ij) + b_i + ε_ij

  f(X_ij)  — XGBoost fixed-effects function: nonlinear function of pitch
              features (location, movement, velocity, intercepts). This is
              what any pitcher would induce on an *average* batter.
  b_i      — Gaussian random intercept per batter, estimated by GLS:
              b_i ~ N(0, σ_b²). This is the batter's persistent tendency
              to place the barrel better or worse than pitch features predict.
  ε_ij     — i.i.d. residual noise: ε ~ N(0, σ²).

The EM algorithm alternates between:
  E-step: update batter random effects given current f()
  M-step: refit XGBoost on (y - b_i) to update f()

Fixed-effects features
----------------------
  release_speed_c     — pitch velocity, z-scored
  plate_x_bat_flip    — horizontal location flipped for RHB (positive = outer edge)
  plate_z             — vertical location
  intercept_y         — depth at which ball crosses batter's frontal plane
  pitch_type dummies  — one-hot encoded pitch type (added automatically)

Random effects grouping
-----------------------
  batter              — one random intercept per batter (scalar b_i)
  b_i > 0 → batter has larger barrel_distance than pitch features predict (worse)
  b_i < 0 → batter has smaller barrel_distance than pitch features predict (better)

  To group by batter:pitch_type instead, change CLUSTER_COL to
  'batter_pitch_type'. More groups → sparser data per group → noisier effects.

Modifying this model
---------------------
FEATURES        list below — add/remove pitch features freely.
                All numeric; pitch_type is dummified automatically.
                pfx variables are currently commented out — uncomment to include.

CLUSTER_COL     which column defines the random-effect grouping.
                'batter' (current) → one intercept per batter.
                'batter_pitch_type' → one intercept per batter×pitch_type cell.

XGB_PARAMS      standard XGBoost hyperparameters. Key ones to tune:
                  n_estimators  — more trees = more capacity, slower
                  max_depth     — tree depth; 4-6 usually good for tabular data
                  learning_rate — smaller = needs more trees but generalises better
                  subsample / colsample_bytree — row/column subsampling for
                                                 regularisation
                  min_child_weight — minimum observations per leaf; raise to
                                     reduce overfitting on rare pitch types

MAX_ITER        MERF EM iterations. 10-20 usually sufficient; convergence is
                monitored via GLL (generalised log-likelihood).

MIN_PA          minimum AB proxy to include a batter in the scoring output.

BATTER_AGG_FEATURES
                Optional batter-level aggregate features to help XGBoost
                approximate random slopes. Set to [] to disable.

Outputs
-------
  barrel_dist_merf_fitted.csv       fitted values + residuals for all obs
  barrel_dist_merf_scores.csv       per-batter scores (≥ MIN_PA)
  barrel_dist_merf_importance.png   XGBoost feature importance
  barrel_dist_merf_diagnostics.png  residual diagnostic plots
  barrel_dist_merf_re.csv           estimated random intercepts per batter

Requires: merf, xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn
  pip install merf xgboost scikit-learn
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from merf import MERF
from time import time

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS      = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL      = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY   = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS

# ── Model configuration ────────────────────────────────────────────────────────

# Fixed-effects features fed to XGBoost.
# pitch_type_* dummies are added automatically — do not add pitch_type here.
FEATURES = [
    'release_speed_c',    # z-scored pitch velocity
    'plate_x_bat_flip',   # horizontal location, handedness-corrected
    'plate_z',            # vertical location
    # 'abs_pfx_x',          # absolute horizontal break — uncomment to include
    # 'pfx_z',              # vertical break — uncomment to include
    'intercept_y',        # depth at batter frontal plane
]

# Column that defines the random-effect grouping.
# Change to 'batter_pitch_type' to nest within pitch type (more groups, less
# data per group). Anything with >= ~5 observations per group is reasonable.
CLUSTER_COL = 'batter'

# Batter-level aggregate features appended to every row so XGBoost can learn
# batter-specific patterns without batter ID as a raw feature. These approximate
# the random-slopes idea from the lmer model. Set to [] to disable.
BATTER_AGG_FEATURES: list[str] = []

# XGBoost hyperparameters for the fixed-effects learner.
XGB_PARAMS = dict(
    n_estimators       = 300,
    max_depth          = 5,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 20,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    random_state       = 42,
    n_jobs             = -1,
    verbosity          = 0,
)

MAX_ITER = 15    # EM iterations; convergence usually within 10
MIN_PA   = 400   # minimum AB proxy to include batter in scoring output

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'; DARK = '#1F2937'


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # ── Confirm barrel_distance exists ────────────────────────────────────────
    # barrel_distance must be pre-computed by add_barrel_distance.py.
    # It is NaN for non-swing events, so filtering to ALL_SWINGS and then
    # dropping NaN on barrel_distance is equivalent.
    if 'barrel_distance' not in df.columns:
        raise ValueError(
            'barrel_distance column not found. '
            'Run add_barrel_distance.py first.'
        )

    # ── Handedness-corrected plate_x ──────────────────────────────────────────
    # Flip sign for RHB so positive always means toward the outer edge.
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    print('plate_x_bat_flip applied (R→negated, L→unchanged)')

    # ── Absolute pfx_x ────────────────────────────────────────────────────────
    df['abs_pfx_x'] = df['pfx_x'].abs()

    # ── Standardise release_speed ─────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── Rename intercept_y ────────────────────────────────────────────────────
    df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── String types for grouping columns ─────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── AB proxy from full dataset (all pitches, not just swings) ─────────────
    bip_mask = df['description'].isin(IN_PLAY)
    k_mask   = df['events'] == 'strikeout'
    bip_ct   = df[bip_mask].groupby('batter').size().rename('n_bip')
    k_ct     = df[k_mask].groupby('batter').size().rename('n_k')
    ab_proxy = (pd.concat([bip_ct, k_ct], axis=1)
                  .fillna(0).astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])
                  [['ab_proxy']].reset_index())

    # ── Filter to all swing events ────────────────────────────────────────────
    # barrel_distance is NaN on non-swing rows, so dropping NaN on
    # barrel_distance after this filter handles any edge cases cleanly.
    sub = df[df['description'].isin(ALL_SWINGS)].copy()
    print(f'All swing events: {len(sub):,}  '
          f'(miss={sub["description"].isin(MISS).sum():,}  '
          f'foul={sub["description"].isin(FOUL).sum():,}  '
          f'in_play={sub["description"].isin(IN_PLAY).sum():,})')

    # ── Batter-level aggregate features (optional) ────────────────────────────
    if BATTER_AGG_FEATURES:
        agg = (sub.groupby('batter')[BATTER_AGG_FEATURES]
                  .mean()
                  .add_prefix('batter_mean_')
                  .reset_index())
        sub = sub.merge(agg, on='batter', how='left')

    # ── Attach AB proxy ───────────────────────────────────────────────────────
    sub = sub.merge(ab_proxy, on='batter', how='left')
    sub['ab_proxy'] = sub['ab_proxy'].fillna(0)

    # ── batter_pitch_type combo (available if CLUSTER_COL is changed) ─────────
    sub['batter_pitch_type'] = sub['batter'] + '_' + sub['pitch_type'].fillna('UNK')

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)

    # ── Drop rows missing required features or barrel_distance ────────────────
    needed = ['barrel_distance', CLUSTER_COL] + FEATURES
    n_before = len(sub)
    sub = sub.dropna(subset=needed).copy()
    print(f'After dropping NA: {len(sub):,} rows '
          f'(dropped {n_before - len(sub):,}), '
          f'{sub[CLUSTER_COL].nunique():,} unique {CLUSTER_COL} groups')

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MERF INPUTS
# ══════════════════════════════════════════════════════════════════════════════

def build_merf_inputs(sub: pd.DataFrame):
    """
    MERF expects:
      X        — fixed-effects design matrix (n, p)
      Z        — random-effects design matrix (n, q).
                 Column of ones for random intercept only.
                 To add a random slope on e.g. release_speed_c, append that
                 column to Z: np.column_stack([np.ones(len(sub)),
                                               sub['release_speed_c'].values])
      clusters — pandas Series of cluster labels (n,)
      y        — outcome vector (n,)
    """
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols = FEATURES + pt_dummies

    if BATTER_AGG_FEATURES:
        X_cols += [f'batter_mean_{f}' for f in BATTER_AGG_FEATURES
                   if f'batter_mean_{f}' in sub.columns]

    X        = sub[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub), 1))
    clusters = sub[CLUSTER_COL].reset_index(drop=True)
    y        = sub['barrel_distance'].reset_index(drop=True)

    return X, Z, clusters, y, X_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT MERF
# ══════════════════════════════════════════════════════════════════════════════

def fit_merf(X, Z, clusters, y) -> MERF:
    """
    Fit MERF with XGBoost as the fixed-effects learner.

    To swap in a different learner, replace XGBRegressor with any
    sklearn-compatible regressor, e.g.:
      from lightgbm import LGBMRegressor
      learner = LGBMRegressor(n_estimators=300, ...)

    The GLL printed each iteration should increase and plateau at convergence.
    If still climbing at MAX_ITER, increase MAX_ITER.
    """
    xgb = XGBRegressor(**XGB_PARAMS)

    mrf = MERF(
        fixed_effects_model = xgb,
        max_iterations      = MAX_ITER,
    )

    print(f'\nFitting MERF  (cluster={CLUSTER_COL}, '
          f'features={X.shape[1]}, n={len(y):,}, max_iter={MAX_ITER})')
    print('GLL should increase each iteration and plateau at convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS & PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list, out_prefix: str):

    resid = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Barrel Distance – MERF (XGBoost fixed effects)',
                 fontsize=13, fontweight='bold')

    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.10, s=4, color=BLUE, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted barrel_distance'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    # Predicted vs actual
    ax = axes[0, 1]
    lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.scatter(y, fitted, alpha=0.10, s=4, color=BLUE, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y, fitted)[0, 1]
    ax.set_xlabel('Actual barrel_distance'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # GLL convergence
    ax = axes[1, 0]
    gll_hist = mrf.gll_history
    ax.plot(range(1, len(gll_hist) + 1), gll_hist,
            color=GREEN, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('EM Iteration'); ax.set_ylabel('GLL')
    ax.set_title('MERF convergence (GLL per EM iteration)', fontweight='bold')
    ax.grid(alpha=0.3)

    # Residual distribution
    ax = axes[1, 1]
    lo, hi = np.percentile(resid, 0.5), np.percentile(resid, 99.5)
    ax.hist(np.clip(resid, lo, hi), bins=80,
            color=BLUE, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (σ = {resid.std():.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{out_prefix}_diagnostics.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    # Feature importance
    imp = pd.Series(
        mrf.trained_fe_model.feature_importances_,
        index=X_cols
    ).sort_values(ascending=True)

    n = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=GREEN, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title('XGBoost fixed-effects feature importance\n(top 25)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = f'{out_prefix}_importance.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series) -> pd.DataFrame:
    """
    barrel_placement_score = 1 / mean(|residual|) for batters with
    ab_proxy >= MIN_PA.

    Higher score = smaller average unexplained barrel_distance = better
    barrel placement skill after controlling for pitch difficulty.

    Two complementary outputs:
      barrel_placement_score  — within-batter consistency around model prediction
      random_intercept (re)   — batter's persistent baseline shift in
                                barrel_distance; negative = systematically
                                better placement than pitch features predict
    """
    ab_proxy_map = sub.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub['barrel_distance'].values - fitted)

    # Break down event counts per batter for interpretability
    is_miss    = sub['description'].isin(MISS)
    is_contact = sub['description'].isin(FOUL | IN_PLAY)

    event_counts = (
        pd.DataFrame({
            'batter':     clusters.values,
            'is_miss':    is_miss.values,
            'is_contact': is_contact.values,
        })
        .groupby('batter')
        .agg(n_miss=('is_miss', 'sum'), n_contact=('is_contact', 'sum'))
    )

    scores = (
        pd.DataFrame({
            'batter':    clusters.values,
            'abs_resid': resid_abs,
            'ab_proxy':  sub['ab_proxy'].values,
        })
        .groupby('batter')
        .agg(
            n_events         = ('abs_resid', 'size'),
            mean_abs_resid   = ('abs_resid', 'mean'),
            ab_proxy         = ('ab_proxy', 'first'),
        )
        .join(event_counts)
        .loc[lambda df: df.index.isin(qualifying)]
        .assign(barrel_placement_score=lambda df: 1.0 / df['mean_abs_resid'])
        .sort_values('barrel_placement_score', ascending=False)
        .reset_index()
    )

    print(f'\nScores for {len(scores):,} batters (≥{MIN_PA} AB proxy):')
    print(scores[['batter', 'barrel_placement_score',
                  'mean_abs_resid', 'n_events', 'n_miss', 'n_contact']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Barrel Distance – MERF (XGBoost fixed effects)\n')

    sub = prepare_data()

    X, Z, clusters, y, X_cols = build_merf_inputs(sub)

    mrf = fit_merf(X, Z, clusters, y)

    fitted = mrf.predict(X, Z, clusters)

    # ── Overall R² ────────────────────────────────────────────────────────────
    ss_res = np.sum((y.values - fitted) ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f'\nOverall R²: {r2:.4f}')

    # ── Random intercepts ─────────────────────────────────────────────────────
    # b_i < 0 → batter places barrel better than pitch features predict (better)
    # b_i > 0 → batter places barrel worse than pitch features predict (worse)
    re_df = (mrf.trained_b
               .rename(columns={0: 'random_intercept'})
               .reset_index()
               .rename(columns={'index': CLUSTER_COL}))
    re_path = os.path.join(OUT_DIR, 'barrel_dist_merf_re.csv')
    re_df.to_csv(re_path, index=False)
    print(f'\nRandom intercepts → {re_path}')
    print('Best barrel placers (b_i most negative):')
    print(re_df.sort_values('random_intercept').head(5).to_string(index=False))
    print('Worst barrel placers (b_i most positive):')
    print(re_df.sort_values('random_intercept', ascending=False)
               .head(5).to_string(index=False))

    # ── Fitted values CSV ─────────────────────────────────────────────────────
    fitted_df = pd.DataFrame({
        CLUSTER_COL:        clusters.values,
        'description':      sub['description'].values,
        'barrel_distance':  y.values,
        'fitted':           fitted,
        'residual':         y.values - fitted,
    })
    fitted_path = os.path.join(OUT_DIR, 'barrel_dist_merf_fitted.csv')
    fitted_df.to_csv(fitted_path, index=False)
    print(f'Fitted values → {fitted_path}')

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    out_prefix = os.path.join(OUT_DIR, 'barrel_dist_merf')
    diagnostic_plots(y, fitted, mrf, X_cols, out_prefix)

    # ── Scores ────────────────────────────────────────────────────────────────
    scores = compute_scores(sub, fitted, clusters)
    scores_path = os.path.join(OUT_DIR, 'barrel_dist_merf_scores.csv')
    scores.to_csv(scores_path, index=False)
    print(f'Scores → {scores_path}')

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    start = time()
    main()
    elapsed = time()-start
    print(elapsed)
