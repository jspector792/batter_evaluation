"""
barrel_distance_merf_swing_variants.py
=======================================
Three MERF variants predicting barrel_distance_v2, testing whether swing
mechanics features (attack angle, swing path tilt) improve barrel placement
prediction beyond pitch features alone.

barrel_distance_v2 is precomputed by run_frozen_models.py / add_barrel_distance.py
and read directly from the parquet files. No recomputation is needed here.

Base features (all variants)
------------------------------
  release_speed_c     — z-scored pitch velocity
  plate_x_bat_flip    — horizontal location, handedness-corrected
  plate_z             — vertical location
  intercept_y         — depth at batter's frontal plane
  pitch_type dummies  — one-hot encoded

Variant definitions
--------------------
  V1 — base + attack_angle
       Tests whether the batter's attack angle (bat path angle at contact)
       explains barrel placement beyond pitch difficulty alone.
       Expectation: attack angle should help — steeper angles tend to produce
       higher launch angles, shallower angles groundballs. But the relationship
       may not be linear, which is where XGBoost helps.

  V2 — base + swing_path_tilt
       Tests whether swing path tilt (the lateral tilt of the swing plane)
       adds explanatory power. Tilt is more relevant to horizontal barrel
       placement than vertical.

  V3 — base + attack_angle + swing_path_tilt
       Both swing mechanics features together. The interaction between attack
       angle and tilt may matter — a steep attack angle combined with high tilt
       is a very different swing geometry than steep + low tilt.

Random effects
--------------
  One random intercept per batter (b_i ~ N(0, σ_b²)).
  b_i < 0 → batter places barrel better than pitch + swing features predict
  b_i > 0 → batter places barrel worse than pitch + swing features predict

Note on swing features as predictors
--------------------------------------
attack_angle and swing_path_tilt are measurements of *what happened* on each
swing, not purely pitch characteristics. Including them as fixed effects means
the model controls for swing mechanics when estimating batter skill (the random
intercept). This is a more demanding test of skill — the residual represents
barrel placement above expectation given both the pitch AND the swing shape.
Whether this is the right adjustment depends on whether swing mechanics are
themselves a stable skill or a reactive consequence of pitch difficulty.

Column names
------------
  attack_angle      — bat attack angle at contact (degrees); from Statcast
  swing_path_tilt   — lateral tilt of swing plane (degrees); from Statcast
  If your data uses different column names, update ATTACK_ANGLE_COL and
  SWING_TILT_COL below.

Modifying this script
----------------------
BASE_FEATURES       pitch features shared across all variants
ATTACK_ANGLE_COL    raw column name for attack angle
SWING_TILT_COL      raw column name for swing path tilt
CLUSTER_COL         random effects grouping ('batter' or 'batter_pitch_type')
XGB_PARAMS          XGBoost hyperparameters
MAX_ITER            MERF EM iterations
MIN_PA              minimum AB proxy for scoring output

Outputs
-------
  barrel_swing_v1_fitted.csv        fitted values + residuals, V1
  barrel_swing_v2_fitted.csv        fitted values + residuals, V2
  barrel_swing_v3_fitted.csv        fitted values + residuals, V3
  barrel_swing_v1_re.csv            random intercepts, V1
  barrel_swing_v2_re.csv            random intercepts, V2
  barrel_swing_v3_re.csv            random intercepts, V3
  barrel_swing_v1_scores.csv        per-batter scores, V1
  barrel_swing_v2_scores.csv        per-batter scores, V2
  barrel_swing_v3_scores.csv        per-batter scores, V3
  barrel_swing_v1_diagnostics.png   residual plots + GLL, V1
  barrel_swing_v2_diagnostics.png   residual plots + GLL, V2
  barrel_swing_v3_diagnostics.png   residual plots + GLL, V3
  barrel_swing_v1_importance.png    feature importance, V1
  barrel_swing_v2_importance.png    feature importance, V2
  barrel_swing_v3_importance.png    feature importance, V3
  barrel_swing_comparison.csv       side-by-side R² / score correlation table
  barrel_swing_comparison.png       score scatter plots across variants

Requires: merf, xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn
  pip install merf xgboost scikit-learn
"""

import os, glob, warnings
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL  = 'intercept_ball_minus_batter_pos_y_inches'
ATTACK_ANGLE_COL = 'attack_angle'      # update if named differently in your data
SWING_TILT_COL   = 'swing_path_tilt'   # update if named differently in your data
OUTCOME_COL      = 'barrel_distance_v2'

MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS
CONTACT    = IN_PLAY | FOUL

# ── Base features (shared across all variants) ────────────────────────────────
# pitch_type dummies are added automatically — do not add pitch_type here.
BASE_FEATURES = [
    'release_speed_c',   # z-scored pitch velocity
    'plate_x_bat_flip',  # horizontal location, handedness-corrected
    'plate_z',           # vertical location
    'intercept_y',       # depth at batter frontal plane
]

# ── Variant definitions ───────────────────────────────────────────────────────
# Each tuple: (short_label, display_label, extra_features)
VARIANTS = [
    ('v1', 'V1: base + attack_angle',           ['attack_angle']),
    ('v2', 'V2: base + swing_path_tilt',        ['swing_path_tilt']),
    ('v3', 'V3: base + attack_angle + tilt',    ['attack_angle', 'swing_path_tilt']),
]

# ── Clustering ────────────────────────────────────────────────────────────────
CLUSTER_COL = 'batter'

# ── Batter aggregate features ─────────────────────────────────────────────────
# Set to [] to disable. These help XGBoost approximate per-batter slopes.
BATTER_AGG_FEATURES: list[str] = []

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = dict(
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
)

MAX_ITER = 15
MIN_PA   = 400

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE  = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'
VARIANT_COLORS = {'v1': '#7C3AED', 'v2': '#D97706', 'v3': '#0891B2'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # ── Confirm barrel_distance_v2 exists ─────────────────────────────────────
    if OUTCOME_COL not in df.columns:
        raise ValueError(
            f'{OUTCOME_COL!r} column not found. '
            f'Run run_frozen_models.py or add_barrel_distance.py first.'
        )
    print(f'{OUTCOME_COL}: non-null={df[OUTCOME_COL].notna().sum():,}  '
          f'mean={df[OUTCOME_COL].mean():.3f}')

    # ── String types ──────────────────────────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── plate_x_bat_flip ──────────────────────────────────────────────────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    # ── release_speed_c ───────────────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── Rename intercept_y ────────────────────────────────────────────────────
    if INTERCEPT_Y_COL in df.columns:
        df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── Rename swing mechanics columns to canonical names ─────────────────────
    # These are already in the data from Statcast bat tracking.
    # No standardisation applied — they are used on their natural scale
    # so that coefficients and importances are interpretable in degrees.
    if ATTACK_ANGLE_COL != 'attack_angle' and ATTACK_ANGLE_COL in df.columns:
        df = df.rename(columns={ATTACK_ANGLE_COL: 'attack_angle'})
    if SWING_TILT_COL != 'swing_path_tilt' and SWING_TILT_COL in df.columns:
        df = df.rename(columns={SWING_TILT_COL: 'swing_path_tilt'})

    # ── Report swing mechanics availability ───────────────────────────────────
    for col in ['attack_angle', 'swing_path_tilt']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f'{col}: {n:,} non-null ({100*n/len(df):.1f}%)')
        else:
            print(f'WARNING: {col!r} not found in data')

    # ── AB proxy ──────────────────────────────────────────────────────────────
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

    # ── Filter to swing events ────────────────────────────────────────────────
    sub = df[df['description'].isin(ALL_SWINGS)].copy()
    print(f'\nSwing events: {len(sub):,}  '
          f'(miss={sub["description"].isin(MISS).sum():,}  '
          f'foul={sub["description"].isin(FOUL).sum():,}  '
          f'in_play={sub["description"].isin(IN_PLAY).sum():,})')

    # ── Batter aggregate features (optional) ──────────────────────────────────
    if BATTER_AGG_FEATURES:
        available = [f for f in BATTER_AGG_FEATURES if f in sub.columns]
        if available:
            agg = (sub.groupby('batter')[available]
                      .mean().add_prefix('batter_mean_').reset_index())
            sub = sub.merge(agg, on='batter', how='left')

    # ── batter_pitch_type combo ────────────────────────────────────────────────
    sub['batter_pitch_type'] = (
        sub['batter'] + '_' + sub['pitch_type'].fillna('UNK')
    )

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)

    # ── Drop rows missing base features or outcome ────────────────────────────
    # Each variant will additionally drop rows missing its own extra features.
    base_needed = [OUTCOME_COL, CLUSTER_COL] + BASE_FEATURES
    n_before = len(sub)
    sub = sub.dropna(subset=base_needed).copy()
    print(f'After dropping NA on base features: {len(sub):,} rows '
          f'(dropped {n_before - len(sub):,}), '
          f'{sub[CLUSTER_COL].nunique():,} unique batters')

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MERF INPUTS FOR ONE VARIANT
# ══════════════════════════════════════════════════════════════════════════════

def build_inputs(sub: pd.DataFrame,
                 extra_features: list[str]) -> tuple:
    """
    Build MERF inputs for one variant.
    Drops rows missing the variant's extra features on top of the base drop.
    Returns (X, Z, clusters, y, X_cols, sub_v).
    sub_v retains original sub index for score aggregation.
    """
    all_features = BASE_FEATURES + extra_features

    # Drop rows missing extra features for this variant
    n_before = len(sub)
    sub_v = sub.dropna(subset=extra_features).copy()
    n_dropped = n_before - len(sub_v)
    if n_dropped > 0:
        print(f'  Dropped {n_dropped:,} additional rows missing '
              f'{extra_features} → {len(sub_v):,} rows')

    pt_dummies = [c for c in sub_v.columns if c.startswith('pt_')]
    X_cols = all_features + pt_dummies

    if BATTER_AGG_FEATURES:
        X_cols += [f'batter_mean_{f}' for f in BATTER_AGG_FEATURES
                   if f'batter_mean_{f}' in sub_v.columns]

    X        = sub_v[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub_v), 1))
    clusters = sub_v[CLUSTER_COL].reset_index(drop=True)
    y        = sub_v[OUTCOME_COL].reset_index(drop=True)

    return X, Z, clusters, y, X_cols, sub_v


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT MERF
# ══════════════════════════════════════════════════════════════════════════════

def fit_merf(X, Z, clusters, y, label: str) -> MERF:
    xgb = XGBRegressor(**XGB_PARAMS)
    mrf = MERF(fixed_effects_model=xgb, max_iterations=MAX_ITER)

    print(f'\n  Fitting MERF [{label}]  '
          f'(n={len(y):,}, features={X.shape[1]}, max_iter={MAX_ITER})')
    print('  GLL should increase and plateau at convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list,
                     short: str, display_label: str):

    color  = VARIANT_COLORS[short]
    resid  = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Barrel Distance MERF – {display_label}',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.10, s=4, color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted barrel_distance_v2'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.scatter(y, fitted, alpha=0.10, s=4, color=color, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y, fitted)[0, 1]
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    gll = mrf.gll_history
    ax.plot(range(1, len(gll) + 1), gll,
            color=GREEN, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('EM Iteration'); ax.set_ylabel('GLL')
    ax.set_title('MERF convergence', fontweight='bold')
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
    path = os.path.join(OUT_DIR, f'barrel_swing_{short}_diagnostics.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    # Feature importance
    imp = pd.Series(mrf.trained_fe_model.feature_importances_,
                    index=X_cols).sort_values(ascending=True)
    n = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=color, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'XGBoost feature importance – {display_label}\n(top 25)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'barrel_swing_{short}_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    return float(corr)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub_v: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series, short: str) -> pd.DataFrame:

    ab_proxy_map = sub_v.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub_v[OUTCOME_COL].values - fitted)

    is_miss    = sub_v['description'].isin(MISS)
    is_contact = sub_v['description'].isin(CONTACT)

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
            'ab_proxy':  sub_v['ab_proxy'].values,
        })
        .groupby('batter')
        .agg(
            n_events       = ('abs_resid', 'size'),
            mean_abs_resid = ('abs_resid', 'mean'),
            ab_proxy       = ('ab_proxy', 'first'),
        )
        .join(event_counts)
        .loc[lambda df: df.index.isin(qualifying)]
        .assign(barrel_placement_score=lambda df: 1.0 / df['mean_abs_resid'])
        .sort_values('barrel_placement_score', ascending=False)
        .reset_index()
    )

    print(f'\n  Scores for {len(scores):,} batters (≥{MIN_PA} AB proxy):')
    print(scores[['batter', 'barrel_placement_score',
                  'mean_abs_resid', 'n_events', 'n_miss', 'n_contact']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. CROSS-VARIANT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def comparison_plots(all_scores: dict, all_records: list):
    """
    Two comparison outputs:
    1. R² / correlation table across variants
    2. Pairwise score scatter plots (V1 vs V2, V1 vs V3, V2 vs V3)
    """
    # ── Summary table ─────────────────────────────────────────────────────────
    comp_df = pd.DataFrame(all_records)
    comp_path = os.path.join(OUT_DIR, 'barrel_swing_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f'\nVariant comparison:\n{comp_df.to_string(index=False)}')
    print(f'→ {comp_path}')

    # ── Pairwise score scatter plots ──────────────────────────────────────────
    pairs = [('v1', 'v2'), ('v1', 'v3'), ('v2', 'v3')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Barrel placement score: pairwise variant comparison',
                 fontsize=13, fontweight='bold')

    for ax, (a, b) in zip(axes, pairs):
        if a not in all_scores or b not in all_scores:
            continue
        merged = (all_scores[a][['batter', 'barrel_placement_score']]
                  .rename(columns={'barrel_placement_score': f'score_{a}'})
                  .merge(
                      all_scores[b][['batter', 'barrel_placement_score']]
                      .rename(columns={'barrel_placement_score': f'score_{b}'}),
                      on='batter', how='inner'))

        corr = np.corrcoef(merged[f'score_{a}'], merged[f'score_{b}'])[0, 1]
        ax.scatter(merged[f'score_{a}'], merged[f'score_{b}'],
                   alpha=0.4, s=14,
                   color=VARIANT_COLORS[a], rasterized=True)
        lim = [min(merged[f'score_{a}'].min(), merged[f'score_{b}'].min()),
               max(merged[f'score_{a}'].max(), merged[f'score_{b}'].max())]
        ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
        ax.set_xlabel(f'Score {a.upper()}')
        ax.set_ylabel(f'Score {b.upper()}')
        ax.set_title(f'{a.upper()} vs {b.upper()}  (r={corr:.3f})',
                     fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'barrel_swing_comparison.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'→ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Barrel Distance MERF – Swing Mechanics Variants\n')
    print(f'Outcome: {OUTCOME_COL}')
    print(f'Base features: {BASE_FEATURES}')
    print(f'Variants: {[v[1] for v in VARIANTS]}\n')

    sub = prepare_data()

    all_scores  = {}
    all_records = []

    names = ['var_1.joblib','var_2.joblib','var_3.joblib']
    
    dummy=0

    for short, display_label, extra_features in VARIANTS:
        
        print(f'\n{"="*70}')
        print(f'VARIANT: {display_label}')
        print('='*70)

        X, Z, clusters, y, X_cols, sub_v = build_inputs(sub, extra_features)

        mrf    = fit_merf(X, Z, clusters, y, display_label)
        fitted = mrf.predict(X, Z, clusters)
        try:
            joblib.dump(mrf, names[dummy], compress=3)
            dummy+=1
        except:
            print('biffed the save dummy')
        

        # R²
        ss_res = np.sum((y.values - fitted) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f'\n  R²: {r2:.4f}')

        # Random intercepts
        re_df = (mrf.trained_b
                   .rename(columns={0: 'random_intercept'})
                   .reset_index()
                   .rename(columns={'index': CLUSTER_COL}))
        re_path = os.path.join(OUT_DIR, f'barrel_swing_{short}_re.csv')
        re_df.to_csv(re_path, index=False)
        print(f'  Random intercepts → {re_path}')
        print('  Best placers (b_i most negative):')
        print(re_df.sort_values('random_intercept')
                   .head(5).to_string(index=False))
        print('  Worst placers (b_i most positive):')
        print(re_df.sort_values('random_intercept', ascending=False)
                   .head(5).to_string(index=False))

        # Fitted values
        fitted_df = pd.DataFrame({
            CLUSTER_COL:   clusters.values,
            'description': sub_v['description'].values,
            OUTCOME_COL:   y.values,
            'fitted':      fitted,
            'residual':    y.values - fitted,
        })
        fitted_path = os.path.join(OUT_DIR,
                                    f'barrel_swing_{short}_fitted.csv')
        fitted_df.to_csv(fitted_path, index=False)
        print(f'  Fitted values → {fitted_path}')

        # Diagnostics
        corr = diagnostic_plots(y, fitted, mrf, X_cols, short, display_label)

        # Scores
        scores = compute_scores(sub_v, fitted, clusters, short)
        scores_path = os.path.join(OUT_DIR,
                                    f'barrel_swing_{short}_scores.csv')
        scores.to_csv(scores_path, index=False)
        print(f'  Scores → {scores_path}')

        all_scores[short] = scores
        all_records.append({
            'variant':       display_label,
            'n_obs':         len(y),
            'n_features':    X.shape[1],
            'r2':            round(r2, 4),
            'pred_vs_actual_r': round(corr, 4),
            'extra_features': ', '.join(extra_features),
        })

    # ── Cross-variant comparison ───────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('CROSS-VARIANT COMPARISON')
    print('='*70)
    comparison_plots(all_scores, all_records)

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()