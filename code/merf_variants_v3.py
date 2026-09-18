"""
contact_merf_variants.py
========================
Three MERF variants exploring a two-stage contact modelling approach.
All three models fit on contact events only (balls in play + fouls).

The motivation: in the original barrel_distance models, miss events dominate
the outcome distribution and are more predictable (larger distances, clearer
pitch-feature signal). By fitting contact-only, the model focuses entirely
on the harder problem of barrel placement quality given that contact occurred.

Variant 1 — signed barrel distance (contact only)
--------------------------------------------------
  Outcome: barrel_distance_signed = C * sin(la - 20°)
           No absolute value — negative = below ideal angle (groundball side),
           positive = above ideal angle (popup side), zero = perfect 20°.
           This gives a smooth, roughly symmetric distribution centered near
           zero, which is friendlier for MERF's Gaussian assumption than the
           zero-inflated absolute value version.

  Predictors: release_speed_c, plate_x_bat_flip, plate_z, intercept_y,
              attack_angle, swing_path_tilt, pitch_type dummies

  Interpretation: the random intercept b_i captures whether a batter
  systematically hits the ball above or below the ideal launch angle after
  controlling for pitch and swing features.

Variant 2 — launch angle (contact only)
-----------------------------------------
  Outcome: launch_angle (degrees, raw)
  Predictors: same as V1

  Interpretation: a direct model of launch angle. Compared to V1, this is
  on a more interpretable scale and doesn't require the physical collision
  geometry assumption. The random intercept captures each batter's persistent
  launch angle tendency above/below what pitch + swing features predict.

Variant 3 — launch speed (contact only)
-----------------------------------------
  Outcome: launch_speed (exit velocity, mph)
  Predictors: bat_speed_c (replaces release_speed_c), plate_x_bat_flip,
              plate_z, intercept_y, attack_angle, swing_path_tilt,
              pitch_type dummies

  bat_speed_c is z-scored on the contact subset (same approach as
  release_speed_c). Release speed is excluded — bat speed is the primary
  driver of exit velocity, and the two are correlated enough that including
  both adds noise. Release speed can be added back by appending
  'release_speed_c' to V3_FEATURES below.

  Interpretation: the random intercept captures each batter's persistent
  exit velocity tendency above/below what swing + pitch features predict —
  i.e. their ability to transfer energy at contact.

Physical constants
------------------
  C = (bat_diameter/2) + (ball_diameter/2) = 1.305 + 1.450 = 2.755 inches
  LA_CENTER = 20° — ideal launch angle (zero-error point)

Speed settings (trimmed for first-pass testing)
------------------------------------------------
  n_estimators = 150   (vs 300 in previous scripts)
  max_depth    = 4     (vs 5)
  MAX_ITER     = 10    (vs 15)
  Expect ~60% faster runtime with moderate performance reduction.
  Increase these once the approach is validated.

Outputs
-------
  contact_v1_fitted.csv        fitted values, V1
  contact_v2_fitted.csv        fitted values, V2
  contact_v3_fitted.csv        fitted values, V3
  contact_v1_re.csv            random intercepts, V1
  contact_v2_re.csv            random intercepts, V2
  contact_v3_re.csv            random intercepts, V3
  contact_v1_scores.csv        per-batter scores, V1
  contact_v2_scores.csv        per-batter scores, V2
  contact_v3_scores.csv        per-batter scores, V3
  contact_v1_diagnostics.png   residual plots + GLL, V1
  contact_v2_diagnostics.png   residual plots + GLL, V2
  contact_v3_diagnostics.png   residual plots + GLL, V3
  contact_v1_importance.png    feature importance, V1
  contact_v2_importance.png    feature importance, V2
  contact_v3_importance.png    feature importance, V3
  contact_v1_model.joblib      saved model, V1
  contact_v2_model.joblib      saved model, V2
  contact_v3_model.joblib      saved model, V3
  contact_comparison.csv       R² and score correlation table

Requires: merf, xgboost, joblib, scikit-learn, pandas, numpy, matplotlib
  pip install merf xgboost scikit-learn joblib
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

FOUL    = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
CONTACT = IN_PLAY | FOUL

# ── Physical constants ─────────────────────────────────────────────────────────
BAT_DIAMETER  = 2.61
BALL_DIAMETER = 2.90
C         = (BAT_DIAMETER / 2) + (BALL_DIAMETER / 2)   # 2.755 inches
LA_CENTER = 20.0   # degrees

# ── Shared base features (V1 and V2) ──────────────────────────────────────────
V1V2_FEATURES = [
    'release_speed_c',
    'plate_x_bat_flip',
    'plate_z',
    'intercept_y',
    'attack_angle',
    'swing_path_tilt',
]

# ── V3 features (bat_speed_c replaces release_speed_c) ────────────────────────
V3_FEATURES = [
    'bat_speed_c',        # z-scored bat speed, computed on contact subset
    'plate_x_bat_flip',
    'plate_z',
    'intercept_y',
    'attack_angle',
    'swing_path_tilt',
]

CLUSTER_COL = 'batter'
MIN_PA      = 400

# ── XGBoost hyperparameters (trimmed for speed) ────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 300,   # reduced from 300 for speed
    max_depth        = 5,     # reduced from 5 for speed
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

MAX_ITER = 15   # reduced from 15 for speed

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE  = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'
VARIANT_COLORS = {'v1': '#7C3AED', 'v2': '#D97706', 'v3': '#0891B2'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. OUTCOME COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_barrel_distance_signed(df: pd.DataFrame) -> pd.Series:
    """
    Signed barrel distance for contact events only.
      = C * sin(la - 20°)

    No absolute value:
      negative → below ideal angle (groundball/line drive side)
      zero     → exactly 20° (ideal)
      positive → above ideal angle (popup side)

    NaN for non-contact rows or contact rows missing launch_angle.
    """
    bd = pd.Series(np.nan, index=df.index, dtype=float)
    is_contact    = df['description'].isin(CONTACT)
    valid_contact = is_contact & df['launch_angle'].notna()
    la_rad = np.deg2rad(df.loc[valid_contact, 'launch_angle'])
    bd.loc[valid_contact] = C * np.sin(la_rad - np.deg2rad(LA_CENTER))
    return bd


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # ── String types ──────────────────────────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── plate_x_bat_flip ──────────────────────────────────────────────────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    # ── release_speed_c — computed on full dataset before filtering ───────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── intercept_y ───────────────────────────────────────────────────────────
    if INTERCEPT_Y_COL in df.columns:
        df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── Compute signed barrel distance ────────────────────────────────────────
    df['barrel_distance_signed'] = compute_barrel_distance_signed(df)

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

    # ── Filter to contact events only ─────────────────────────────────────────
    sub = df[df['description'].isin(CONTACT)].copy()
    print(f'Contact events: {len(sub):,}  '
          f'(in_play={sub["description"].isin(IN_PLAY).sum():,}  '
          f'foul={sub["description"].isin(FOUL).sum():,})')

    # ── Remove likely bunts and check swings ──────────────────────────────────
    # bat_speed < 50 mph is indicative of intentional bunts or partial swings
    # that are not representative of true contact events.
    n_before = len(sub)
    sub = sub[sub['bat_speed'] >= 50].copy()
    print(f'After bat_speed >= 50 filter: {len(sub):,} '
          f'(removed {n_before - len(sub):,})')

    # ── bat_speed_c — computed on contact subset ──────────────────────────────
    # Computed after contact filter so the mean/sd reflects contact events,
    # matching what V3 will see at prediction time.
    if 'bat_speed' in sub.columns:
        bs_mean = sub['bat_speed'].mean()
        bs_std  = sub['bat_speed'].std()
        sub['bat_speed_c'] = (sub['bat_speed'] - bs_mean) / bs_std
        print(f'bat_speed_c: mean={bs_mean:.2f}, sd={bs_std:.2f}')
    else:
        print('WARNING: bat_speed not found — V3 will fail')
        sub['bat_speed_c'] = np.nan

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)

    # ── batter_pitch_type combo ────────────────────────────────────────────────
    sub['batter_pitch_type'] = (
        sub['batter'] + '_' + sub.get('pitch_type', pd.Series('UNK', index=sub.index))
    )

    print(f'Unique batters: {sub[CLUSTER_COL].nunique():,}')

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MERF INPUTS
# ══════════════════════════════════════════════════════════════════════════════

def build_inputs(sub: pd.DataFrame,
                 features: list[str],
                 outcome_col: str) -> tuple:
    """
    Build MERF inputs for one variant. Drops rows missing the outcome
    or any feature. Returns (X, Z, clusters, y, X_cols, sub_v).
    """
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols     = features + pt_dummies

    needed  = [outcome_col, CLUSTER_COL] + features
    n_before = len(sub)
    sub_v   = sub.dropna(subset=needed).copy()
    print(f'  After dropping NA: {len(sub_v):,} rows '
          f'(dropped {n_before - len(sub_v):,})')

    X        = sub_v[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub_v), 1))
    clusters = sub_v[CLUSTER_COL].reset_index(drop=True)
    y        = sub_v[outcome_col].reset_index(drop=True)

    return X, Z, clusters, y, X_cols, sub_v


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIT MERF
# ══════════════════════════════════════════════════════════════════════════════

def fit_merf(X, Z, clusters, y, label: str) -> MERF:
    xgb = XGBRegressor(**XGB_PARAMS)
    mrf = MERF(fixed_effects_model=xgb, max_iterations=MAX_ITER)

    print(f'\n  Fitting MERF [{label}]  '
          f'(n={len(y):,}, features={X.shape[1]}, '
          f'n_estimators={XGB_PARAMS["n_estimators"]}, '
          f'max_depth={XGB_PARAMS["max_depth"]}, '
          f'max_iter={MAX_ITER})')
    print('  GLL should increase and plateau at convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 5. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list,
                     short: str, display_label: str, outcome_col: str):

    color = VARIANT_COLORS[short]
    resid = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Contact MERF – {display_label}',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.10, s=4, color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel(f'Fitted {outcome_col}'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.scatter(y, fitted, alpha=0.10, s=4, color=color, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y, fitted)[0, 1]
    ax.set_xlabel(f'Actual {outcome_col}'); ax.set_ylabel('Predicted')
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
    path = os.path.join(OUT_DIR, f'contact_{short}_diagnostics.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

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
    path = os.path.join(OUT_DIR, f'contact_{short}_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    return float(corr)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub_v: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series,
                   outcome_col: str, short: str) -> pd.DataFrame:
    """
    score = 1 / mean(|residual|) for batters with ab_proxy >= MIN_PA.

    For V1 (signed barrel distance): lower absolute residual = more
    consistently hitting the ideal launch angle.
    For V2 (launch angle): lower absolute residual = more consistently
    achieving expected launch angle given pitch + swing features.
    For V3 (launch speed): lower absolute residual = more consistently
    achieving expected exit velocity.
    """
    ab_proxy_map = sub_v.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub_v[outcome_col].values - fitted)

    scores = (
        pd.DataFrame({
            'batter':    clusters.values,
            'abs_resid': resid_abs,
            'ab_proxy':  sub_v['ab_proxy'].values,
        })
        .groupby('batter')
        .agg(
            n_contact      = ('abs_resid', 'size'),
            mean_abs_resid = ('abs_resid', 'mean'),
            ab_proxy       = ('ab_proxy', 'first'),
        )
        .loc[lambda df: df.index.isin(qualifying)]
        .assign(score=lambda df: 1.0 / df['mean_abs_resid'])
        .rename(columns={'score': f'{short}_score'})
        .sort_values(f'{short}_score', ascending=False)
        .reset_index()
    )

    print(f'\n  Scores for {len(scores):,} batters (≥{MIN_PA} AB proxy):')
    print(scores[['batter', f'{short}_score', 'mean_abs_resid', 'n_contact']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Contact MERF Variants\n')
    print(f'C = {C:.4f} inches, LA_CENTER = {LA_CENTER}°')
    print(f'XGB: n_estimators={XGB_PARAMS["n_estimators"]}, '
          f'max_depth={XGB_PARAMS["max_depth"]}, MAX_ITER={MAX_ITER}\n')

    sub = prepare_data()

    variants = [
        dict(
            short         = 'v1',
            display_label = 'V1: signed barrel distance (contact only)',
            outcome_col   = 'barrel_distance_signed',
            features      = V1V2_FEATURES,
        ),
        dict(
            short         = 'v3',
            display_label = 'V3: launch speed / bat speed (contact only)',
            outcome_col   = 'launch_speed',
            features      = V3_FEATURES,
        ),
    ]

    all_scores  = {}
    all_records = []

    for cfg in variants:
        short         = cfg['short']
        display_label = cfg['display_label']
        outcome_col   = cfg['outcome_col']
        features      = cfg['features']

        print(f'\n{"="*70}')
        print(f'VARIANT: {display_label}')
        print(f'Outcome: {outcome_col}')
        print('='*70)

        X, Z, clusters, y, X_cols, sub_v = build_inputs(
            sub, features, outcome_col)

        mrf    = fit_merf(X, Z, clusters, y, display_label)
        fitted = mrf.predict(X, Z, clusters)

        # R²
        ss_res = np.sum((y.values - fitted) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f'\n  R²: {r2:.4f}')

        # Save model
        model_path = os.path.join(OUT_DIR, f'contact_{short}_model.joblib')
        joblib.dump(mrf, model_path, compress=3)
        print(f'  Model saved → {model_path}')

        # Random intercepts
        re_df = (mrf.trained_b
                   .rename(columns={0: 'random_intercept'})
                   .reset_index()
                   .rename(columns={'index': CLUSTER_COL}))
        re_path = os.path.join(OUT_DIR, f'contact_{short}_re.csv')
        re_df.to_csv(re_path, index=False)
        print(f'  Random intercepts → {re_path}')
        print('  Best (b_i most negative):')
        print(re_df.sort_values('random_intercept')
                   .head(5).to_string(index=False))
        print('  Worst (b_i most positive):')
        print(re_df.sort_values('random_intercept', ascending=False)
                   .head(5).to_string(index=False))

        # Fitted values
        fitted_series = pd.Series(fitted, index=sub_v.index)
        fitted_df = pd.DataFrame({
            CLUSTER_COL:   clusters.values,
            'description': sub_v['description'].values,
            'launch_angle': sub_v['launch_angle'].values,
            outcome_col:   y.values,
            'fitted':      fitted,
            'residual':    y.values - fitted,
        })
        fitted_path = os.path.join(OUT_DIR, f'contact_{short}_fitted.csv')
        fitted_df.to_csv(fitted_path, index=False)
        print(f'  Fitted values → {fitted_path}')

        # Diagnostics
        corr = diagnostic_plots(
            y, fitted, mrf, X_cols, short, display_label, outcome_col)

        # Scores
        scores = compute_scores(sub_v, fitted, clusters, outcome_col, short)
        scores_path = os.path.join(OUT_DIR, f'contact_{short}_scores.csv')
        scores.to_csv(scores_path, index=False)
        print(f'  Scores → {scores_path}')

        all_scores[short] = scores
        all_records.append({
            'variant':       display_label,
            'outcome':       outcome_col,
            'n_obs':         len(y),
            'n_features':    X.shape[1],
            'r2':            round(r2, 4),
            'pred_vs_actual_r': round(corr, 4),
        })

    # ── Comparison table ───────────────────────────────────────────────────────
    comp_df = pd.DataFrame(all_records)
    comp_path = os.path.join(OUT_DIR, 'contact_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f'\nVariant comparison:\n{comp_df.to_string(index=False)}')
    print(f'→ {comp_path}')

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()