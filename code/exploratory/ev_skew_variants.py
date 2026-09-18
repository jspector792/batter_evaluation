"""
ev_merf_variants.py
===================
Two MERF variants for predicting exit velocity (launch_speed) on contact
events, addressing the skewed EV distribution identified in earlier modelling.

Background
----------
The baseline contact V3 model (contact_merf_variants.py) predicts launch_speed
directly with a standard squared-error XGBoost objective. Exit velocity has a
left-skewed distribution — the bulk of contact is in the 85-105 mph range with
a long left tail from mishits. This violates MERF's Gaussian residual assumption
and produces poor fit for weak contact events.

Two remedies are tested here:

Variant 1: Reflected log transformation
-----------------------------------------
  Instead of predicting launch_speed directly, we predict:

    y_transformed = log(MAX_EV - launch_speed + OFFSET)

  where MAX_EV = 120 mph (above the observed maximum) and OFFSET = 1 to
  avoid log(0). This reflects and compresses the left tail, producing a
  more symmetric distribution that better satisfies the Gaussian assumption.

  Predictions are back-transformed for scoring and output:
    launch_speed_pred = MAX_EV + OFFSET - exp(fitted)

  The score is computed on the back-transformed scale (mph) so it remains
  interpretable. Residuals are also reported in mph.

  Note: back-transformed predictions will not exactly reproduce the original
  scale's mean due to Jensen's inequality, but the ranking of batters should
  be consistent.

Variant 2: Huber loss
----------------------
  Predict launch_speed directly (no transformation), but replace the
  squared-error XGBoost objective with the Huber loss:

    L(r) = r²/2              if |r| ≤ delta
           delta*(|r| - delta/2)  if |r| > delta

  The Huber loss is quadratic near zero (like MSE) but linear for large
  residuals (like MAE), making it robust to the outliers and skew in the
  left tail without requiring a transformation. The MERF GLS step still
  assumes Gaussian random effects — this only changes how the fixed-effects
  XGBoost handles the skew in the M-step.

  delta (the Huber transition point) is set to 1 SD of launch_speed by
  default. Increase HUBER_DELTA_MULTIPLIER to be more aggressive about
  downweighting outliers, decrease it to be more conservative.

Predictors (same as contact V3 baseline)
-----------------------------------------
  bat_speed_c       — z-scored bat speed (on contact subset)
  plate_x_bat_flip  — horizontal location, handedness-corrected
  plate_z           — vertical location
  intercept_y       — depth at batter frontal plane
  attack_angle      — bat attack angle at contact
  swing_path_tilt   — lateral tilt of swing plane
  pitch_type dummies

Random effects
--------------
  One random intercept per batter (b_i ~ N(0, σ_b²)).
  b_i < 0 → batter generates less EV than pitch + swing features predict
  b_i > 0 → batter generates more EV than pitch + swing features predict

Comparison outputs
------------------
  Residual distribution plots for each variant and the V3 baseline
  (if contact_v3_fitted.csv exists) to visually assess improvement.

Modifying this script
----------------------
  MAX_EV                   upper bound for reflection transform (mph)
  OFFSET                   prevents log(0) in transform
  HUBER_DELTA_MULTIPLIER   sets Huber delta as N * sd(launch_speed)
  XGB_PARAMS               shared XGBoost hyperparameters
  MAX_ITER                 MERF EM iterations
  MIN_PA                   minimum AB proxy for scoring

Outputs
-------
  ev_variant1_fitted.csv        fitted + back-transformed values, V1
  ev_variant2_fitted.csv        fitted values, V2
  ev_variant1_re.csv            random intercepts, V1
  ev_variant2_re.csv            random intercepts, V2
  ev_variant1_scores.csv        per-batter scores (mph residual), V1
  ev_variant2_scores.csv        per-batter scores, V2
  ev_variant1_model.joblib      saved model, V1
  ev_variant2_model.joblib      saved model, V2
  ev_variant1_diagnostics.png   residual plots + GLL, V1
  ev_variant2_diagnostics.png   residual plots + GLL, V2
  ev_variant1_importance.png    feature importance, V1
  ev_variant2_importance.png    feature importance, V2
  ev_variants_comparison.png    residual distribution comparison
  ev_variants_comparison.csv    R² and score summary

Requires: merf, xgboost, joblib, pandas, numpy, matplotlib, seaborn
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from xgboost import XGBRegressor
from merf import MERF

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'all_pitches_2025')
OUT_DIR  = os.path.join(BASE_DIR, 'out', 'exploratory', 'final_models_variants')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

FOUL    = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
CONTACT = IN_PLAY | FOUL

OUTCOME_COL = 'launch_speed'

# ── Transformation constants ───────────────────────────────────────────────────
MAX_EV = 120.0   # mph — above observed max; used as reflection point
OFFSET =   1.0   # prevents log(0); launch_speed < MAX_EV always, so
                  # MAX_EV - launch_speed + OFFSET >= 1

# ── Huber loss ─────────────────────────────────────────────────────────────────
# delta = HUBER_DELTA_MULTIPLIER * sd(launch_speed on contact subset)
# 1.0 = standard choice; decrease toward 0.5 to downweight outliers more
HUBER_DELTA_MULTIPLIER = 1.0

# ── Features (same as contact V3) ─────────────────────────────────────────────
FEATURES = [
    'bat_speed_c',
    'plate_x_bat_flip',
    'plate_z',
    'intercept_y',
    'attack_angle',
    'swing_path_tilt',
]

CLUSTER_COL = 'batter'
MIN_PA      = 400
BAT_SPEED_MIN = 50.0   # filter bunts/check swings (same as contact_merf_variants)

# ── XGBoost base params (objective set per variant) ───────────────────────────
XGB_BASE = dict(
    n_estimators     = 150,
    max_depth        = 4,
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

MAX_ITER = 10

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE   = '#2563EB'; RED  = '#DC2626'; GREEN  = '#16A34A'
GRAY   = '#6B7280'; PURPLE = '#7C3AED'; ORANGE = '#D97706'
VARIANT_COLORS = {'v1': PURPLE, 'v2': ORANGE}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANSFORMATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# def transform_ev(ev: pd.Series) -> pd.Series:
#     """
#     Reflected log transform. Clamps to avoid log(0) or log(negative).
#     """
#     inner = MAX_EV - ev + OFFSET
#     inner = inner.clip(lower=1e-6)   # prevents log(0) or log(negative)
#     return np.log(inner)


# def back_transform_ev(y_transformed: np.ndarray) -> np.ndarray:
#     """
#     Inverse: MAX_EV + OFFSET - exp(y_transformed).
#     """
#     return MAX_EV + OFFSET - np.exp(y_transformed)

def transform_ev(ev: pd.Series) -> pd.Series:
    """Reflected square root: sqrt(MAX_EV - ev + OFFSET)"""
    return np.sqrt((MAX_EV - ev + OFFSET).clip(lower=0))

def back_transform_ev(y_transformed: np.ndarray) -> np.ndarray:
    return MAX_EV + OFFSET - y_transformed ** 2


def check_transform(ev: pd.Series):
    """Print a quick diagnostic showing the transform's effect on skew."""
    transformed = transform_ev(ev)
    print(f'\n  Raw EV:          mean={ev.mean():.2f}  '
          f'sd={ev.std():.2f}  '
          f'skew={ev.skew():.3f}')
    print(f'  Transformed EV:  mean={transformed.mean():.3f}  '
          f'sd={transformed.std():.3f}  '
          f'skew={transformed.skew():.3f}')
    print(f'  (negative skew closer to 0 = more symmetric)')


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

    # ── release_speed_c (full dataset, for consistency) ───────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std

    # ── intercept_y ───────────────────────────────────────────────────────────
    if INTERCEPT_Y_COL in df.columns:
        df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

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

    # ── Filter to contact events ───────────────────────────────────────────────
    sub = df[df['description'].isin(CONTACT)].copy()
    print(f'Contact events: {len(sub):,}')

    # ── Remove bunts / check swings ───────────────────────────────────────────
    n_before = len(sub)
    sub = sub[sub['bat_speed'] >= BAT_SPEED_MIN].copy()
    print(f'After bat_speed >= {BAT_SPEED_MIN}: {len(sub):,} '
          f'(removed {n_before - len(sub):,})')

    # ── bat_speed_c (contact subset) ──────────────────────────────────────────
    bs_mean = sub['bat_speed'].mean()
    bs_std  = sub['bat_speed'].std()
    sub['bat_speed_c'] = (sub['bat_speed'] - bs_mean) / bs_std
    print(f'bat_speed_c: mean={bs_mean:.2f}, sd={bs_std:.2f}')

    # ── Huber delta ───────────────────────────────────────────────────────────
    ev_sd = sub[OUTCOME_COL].dropna().std()
    huber_delta = HUBER_DELTA_MULTIPLIER * ev_sd
    print(f'Huber delta: {huber_delta:.3f} mph '
          f'(= {HUBER_DELTA_MULTIPLIER} × SD={ev_sd:.3f})')

    # ── Check transform skew improvement ──────────────────────────────────────
    check_transform(sub[OUTCOME_COL].dropna())

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt',
                         drop_first=False)

    # ── Drop rows missing outcome or features ─────────────────────────────────
    needed = [OUTCOME_COL, CLUSTER_COL] + FEATURES
    n_before = len(sub)
    sub = sub.dropna(subset=needed).copy()
    print(f'After dropping NA: {len(sub):,} rows '
          f'(dropped {n_before - len(sub):,}), '
          f'{sub[CLUSTER_COL].nunique():,} batters')

    return sub, huber_delta


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MERF INPUTS
# ══════════════════════════════════════════════════════════════════════════════

def build_inputs(sub: pd.DataFrame,
                 outcome_col: str) -> tuple:
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols     = FEATURES + pt_dummies

    X        = sub[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub), 1))
    clusters = sub[CLUSTER_COL].reset_index(drop=True)
    y        = sub[outcome_col].reset_index(drop=True)

    return X, Z, clusters, y, X_cols


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIT MERF
# ══════════════════════════════════════════════════════════════════════════════

def fit_merf(X, Z, clusters, y,
             xgb_params: dict, label: str) -> MERF:
    xgb = XGBRegressor(**xgb_params)
    mrf = MERF(fixed_effects_model=xgb, max_iterations=MAX_ITER)

    print(f'\n  Fitting MERF [{label}]  '
          f'(n={len(y):,}, features={X.shape[1]}, '
          f'max_iter={MAX_ITER}, '
          f'objective={xgb_params.get("objective", "reg:squarederror")})')
    print('  GLL should increase and plateau at convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 5. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y_raw: pd.Series,
                     fitted_raw: np.ndarray,
                     mrf: MERF,
                     X_cols: list,
                     short: str,
                     display_label: str):
    """
    y_raw and fitted_raw are always on the RAW mph scale
    (back-transformed for V1) so residuals are interpretable.
    """
    color = VARIANT_COLORS[short]
    resid = y_raw.values - fitted_raw

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Exit Velocity MERF – {display_label}',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted_raw, resid, alpha=0.10, s=4,
               color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted EV (mph)'); ax.set_ylabel('Residual (mph)')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    lim = [min(y_raw.min(), fitted_raw.min()),
           max(y_raw.max(), fitted_raw.max())]
    ax.scatter(y_raw, fitted_raw, alpha=0.10, s=4,
               color=color, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y_raw, fitted_raw)[0, 1]
    ax.set_xlabel('Actual EV (mph)'); ax.set_ylabel('Predicted EV (mph)')
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

    # Overlay normal reference
    x_ref = np.linspace(lo, hi, 200)
    pdf   = stats.norm.pdf(x_ref, resid.mean(), resid.std())
    ax2   = ax.twinx()
    ax2.plot(x_ref, pdf, color=GRAY, linewidth=1.5,
             linestyle='--', alpha=0.7, label='Normal ref')
    ax2.set_ylabel('Density', color=GRAY, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=GRAY)

    skewness = stats.skew(resid)
    ax.set_xlabel('Residual (mph)'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (σ={resid.std():.2f}, skew={skewness:.3f})',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'ev_variant{short[-1]}_diagnostics.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    # Feature importance
    imp = pd.Series(mrf.trained_fe_model.feature_importances_,
                    index=X_cols).sort_values(ascending=True)
    n   = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=color, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'XGBoost feature importance – {display_label}\n(top 25)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'ev_variant{short[-1]}_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    return float(corr), float(skewness)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub: pd.DataFrame,
                   y_raw: pd.Series,
                   fitted_raw: np.ndarray,
                   clusters: pd.Series,
                   short: str) -> pd.DataFrame:
    """
    Scores are always on the raw mph scale regardless of variant.
    score = 1 / mean(|residual_mph|) for batters with ab_proxy >= MIN_PA.
    Higher = smaller unexplained EV deviation = more consistent hard contact.
    """
    ab_proxy_map = sub.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(y_raw.values - fitted_raw)

    scores = (
        pd.DataFrame({
            'batter':    clusters.values,
            'abs_resid': resid_abs,
            'ab_proxy':  sub['ab_proxy'].values,
        })
        .groupby('batter')
        .agg(
            n_contact      = ('abs_resid', 'size'),
            mean_abs_resid = ('abs_resid', 'mean'),
            ab_proxy       = ('ab_proxy', 'first'),
        )
        .loc[lambda df: df.index.isin(qualifying)]
        .assign(**{f'ev_score_{short}': lambda df: 1.0 / df['mean_abs_resid']})
        .sort_values(f'ev_score_{short}', ascending=False)
        .reset_index()
    )

    print(f'\n  [{short}] Scores for {len(scores):,} batters (≥{MIN_PA} PA):')
    print(scores[['batter', f'ev_score_{short}',
                  'mean_abs_resid', 'n_contact']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════

def comparison_plot(records: list, fitted_data: dict):
    """
    Three-panel comparison:
      Left:   R² and residual skew bar chart across variants (+ baseline if available)
      Middle: residual distribution overlay (V1 vs V2)
      Right:  score scatter V1 vs V2
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('EV variant comparison: transform vs Huber loss',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: R² and skew ──────────────────────────────────────────────────
    ax  = axes[0]
    df  = pd.DataFrame(records)
    x   = np.arange(len(df))
    ax.bar(x - 0.2, df['r2'],   width=0.35, color=BLUE,
           label='R² (raw scale)', alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, df['resid_skew'].abs(), width=0.35,
            color=ORANGE, label='|Residual skew|', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], fontsize=10)
    ax.set_ylabel('R²', color=BLUE)
    ax2.set_ylabel('|Residual skew| (lower = better)', color=ORANGE)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax.set_title('R² and residual skew\n(skew closer to 0 = Gaussian assumption better met)',
                 fontweight='bold', fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: residual distribution overlay ────────────────────────────────
    ax = axes[1]
    for short, label, color in [
        ('v1', 'V1: transform', PURPLE),
        ('v2', 'V2: Huber',     ORANGE),
    ]:
        if short in fitted_data:
            resid = fitted_data[short]['resid']
            lo    = np.percentile(resid, 0.5)
            hi    = np.percentile(resid, 99.5)
            ax.hist(np.clip(resid, lo, hi), bins=60,
                    color=color, alpha=0.5, label=label,
                    edgecolor='none', density=True)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual (mph)')
    ax.set_ylabel('Density')
    ax.set_title('Residual distributions\n(V1 vs V2)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 3: score scatter ────────────────────────────────────────────────
    ax = axes[2]
    if 'v1' in fitted_data and 'v2' in fitted_data:
        s1 = fitted_data['v1']['scores']
        s2 = fitted_data['v2']['scores']
        merged = (s1[['batter', 'ev_score_v1']]
                  .merge(s2[['batter', 'ev_score_v2']],
                         on='batter', how='inner'))
        if len(merged) > 5:
            corr = np.corrcoef(merged['ev_score_v1'],
                               merged['ev_score_v2'])[0, 1]
            ax.scatter(merged['ev_score_v1'], merged['ev_score_v2'],
                       s=14, alpha=0.5, color=BLUE, rasterized=True)
            lim = [min(merged['ev_score_v1'].min(),
                       merged['ev_score_v2'].min()),
                   max(merged['ev_score_v1'].max(),
                       merged['ev_score_v2'].max())]
            ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
            ax.set_xlabel('EV score V1 (transform)', fontsize=10)
            ax.set_ylabel('EV score V2 (Huber)', fontsize=10)
            ax.set_title(f'Score agreement\n(r={corr:.3f}, n={len(merged)})',
                         fontweight='bold', fontsize=10)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'ev_variants_comparison.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('EV MERF Variants — Transform vs Huber Loss\n')

    sub, huber_delta = prepare_data()

    records     = []
    fitted_data = {}

    # ── Variant 1: Reflected log transform ────────────────────────────────────
    print(f'\n{"="*70}')
    print('VARIANT V1: Reflected log transformation')
    print('='*70)

    sub['ev_transformed'] = transform_ev(sub[OUTCOME_COL])
    print(f'  Outcome: ev_transformed = log({MAX_EV} - launch_speed + {OFFSET})')

    # Guard against inf/nan in transformed outcome
    bad_mask = ~np.isfinite(sub['ev_transformed'])
    if bad_mask.sum() > 0:
        print(f'  WARNING: {bad_mask.sum()} inf/nan values in ev_transformed '
              f'(launch_speed range: {sub.loc[bad_mask, OUTCOME_COL].describe()})')
        sub = sub[~bad_mask].copy()
        print(f'  Dropped bad rows → {len(sub):,} remaining')

    X, Z, clusters, y_t, X_cols = build_inputs(sub, 'ev_transformed')

    xgb_params_v1 = {**XGB_BASE, 'objective': 'reg:squarederror'}
    mrf_v1 = fit_merf(X, Z, clusters, y_t, xgb_params_v1,
                      'V1: transform')

    fitted_t  = mrf_v1.predict(X, Z, clusters)
    fitted_v1 = back_transform_ev(fitted_t)         # back to mph
    y_raw_v1  = pd.Series(sub[OUTCOME_COL].values,
                           index=range(len(sub)))

    ss_res = np.sum((y_raw_v1.values - fitted_v1) ** 2)
    ss_tot = np.sum((y_raw_v1.values - y_raw_v1.mean()) ** 2)
    r2_v1  = 1 - ss_res / ss_tot
    print(f'\n  R² (raw mph scale): {r2_v1:.4f}')

    corr_v1, skew_v1 = diagnostic_plots(
        y_raw_v1, fitted_v1, mrf_v1, X_cols, 'v1',
        'V1: reflected log transform')

    joblib.dump(mrf_v1, os.path.join(OUT_DIR, 'ev_variant1_model.joblib'),
                compress=3)

    re_v1 = (mrf_v1.trained_b
               .rename(columns={0: 'random_intercept'})
               .reset_index().rename(columns={'index': CLUSTER_COL}))
    re_v1.to_csv(os.path.join(OUT_DIR, 'ev_variant1_re.csv'), index=False)
    print(f'  Best EV generators (b_i most positive):')
    print(re_v1.sort_values('random_intercept', ascending=False)
               .head(5).to_string(index=False))

    scores_v1 = compute_scores(sub, y_raw_v1, fitted_v1, clusters, 'v1')
    scores_v1.to_csv(os.path.join(OUT_DIR, 'ev_variant1_scores.csv'),
                     index=False)

    fitted_df_v1 = pd.DataFrame({
        CLUSTER_COL:       clusters.values,
        'description':     sub['description'].values,
        'launch_angle':    sub['launch_angle'].values if 'launch_angle' in sub.columns else np.nan,
        OUTCOME_COL:       y_raw_v1.values,
        'ev_transformed':  y_t.values,
        'fitted_transformed': fitted_t,
        'fitted_mph':      fitted_v1,
        'residual_mph':    y_raw_v1.values - fitted_v1,
    })
    fitted_df_v1.to_csv(os.path.join(OUT_DIR, 'ev_variant1_fitted.csv'),
                         index=False)

    records.append(dict(label='V1: transform', r2=round(r2_v1, 4),
                        pred_r=round(corr_v1, 4),
                        resid_skew=round(skew_v1, 4)))
    fitted_data['v1'] = {
        'resid':  y_raw_v1.values - fitted_v1,
        'scores': scores_v1,
    }

    # ── Variant 2: Huber loss ─────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'VARIANT V2: Huber loss (delta={huber_delta:.3f} mph)')
    print('='*70)

    X, Z, clusters, y_raw_v2, X_cols = build_inputs(sub, OUTCOME_COL)

    xgb_params_v2 = {
        **XGB_BASE,
        'objective': 'reg:pseudohubererror',
        # XGBoost uses huber_slope to set the transition point.
        # Smaller huber_slope = more robust (more linear in tails).
        # We pass it via the params dict; XGBoost >= 1.6 supports this.
        'huber_slope': huber_delta,
    }
    mrf_v2 = fit_merf(X, Z, clusters, y_raw_v2, xgb_params_v2,
                      'V2: Huber loss')

    fitted_v2 = mrf_v2.predict(X, Z, clusters)

    ss_res = np.sum((y_raw_v2.values - fitted_v2) ** 2)
    ss_tot = np.sum((y_raw_v2.values - y_raw_v2.mean()) ** 2)
    r2_v2  = 1 - ss_res / ss_tot
    print(f'\n  R² (raw mph scale): {r2_v2:.4f}')

    corr_v2, skew_v2 = diagnostic_plots(
        y_raw_v2, fitted_v2, mrf_v2, X_cols, 'v2',
        f'V2: Huber loss (δ={huber_delta:.2f} mph)')

    joblib.dump(mrf_v2, os.path.join(OUT_DIR, 'ev_variant2_model.joblib'),
                compress=3)

    re_v2 = (mrf_v2.trained_b
               .rename(columns={0: 'random_intercept'})
               .reset_index().rename(columns={'index': CLUSTER_COL}))
    re_v2.to_csv(os.path.join(OUT_DIR, 'ev_variant2_re.csv'), index=False)
    print(f'  Best EV generators (b_i most positive):')
    print(re_v2.sort_values('random_intercept', ascending=False)
               .head(5).to_string(index=False))

    scores_v2 = compute_scores(sub, y_raw_v2, fitted_v2, clusters, 'v2')
    scores_v2.to_csv(os.path.join(OUT_DIR, 'ev_variant2_scores.csv'),
                     index=False)

    fitted_df_v2 = pd.DataFrame({
        CLUSTER_COL:    clusters.values,
        'description':  sub['description'].values,
        'launch_angle': sub['launch_angle'].values if 'launch_angle' in sub.columns else np.nan,
        OUTCOME_COL:    y_raw_v2.values,
        'fitted_mph':   fitted_v2,
        'residual_mph': y_raw_v2.values - fitted_v2,
    })
    fitted_df_v2.to_csv(os.path.join(OUT_DIR, 'ev_variant2_fitted.csv'),
                         index=False)

    records.append(dict(label='V2: Huber', r2=round(r2_v2, 4),
                        pred_r=round(corr_v2, 4),
                        resid_skew=round(skew_v2, 4)))
    fitted_data['v2'] = {
        'resid':  y_raw_v2.values - fitted_v2,
        'scores': scores_v2,
    }

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('COMPARISON')
    print('='*70)

    comp_df = pd.DataFrame(records)
    comp_path = os.path.join(OUT_DIR, 'ev_variants_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f'\n{comp_df.to_string(index=False)}')
    print(f'\n  Lower |resid_skew| = residual distribution closer to Gaussian')
    print(f'  → better satisfied MERF assumption, more reliable random intercepts')

    comparison_plot(records, fitted_data)

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()