"""
barrel_distance_merf_variants.py
=================================
Two MERF variants with alternative barrel_distance outcome definitions,
compared against each other. Neither requires modifying the parquet files —
barrel_distance is computed in memory for each variant.

Variant 1: contact = zero
--------------------------
  swing and miss  → miss_distance + C      (same as original)
  any contact     → 0                      (all contact treated as perfect)

  Interpretation: rewards contact itself, ignoring quality. Equivalent to
  asking "does this batter make contact where the model expects a miss?"
  Warning: produces a zero-inflated outcome (~50-60% zeros on typical contact
  rates). MERF's GLS step assumes Gaussian residuals so convergence may be
  slower and random intercept estimates slightly attenuated. Watch GLL trace.

Variant 2: single-boundary, 20° centered
------------------------------------------
  swing and miss  → miss_distance + C      (same as original)
  any contact     → |C * sin(la - 20°)|

  Interpretation: 20° is the only zero-error point. All other contact angles
  are penalized by their angular distance from 20°, continuously and
  symmetrically. No flat reward region. Penalty at barrel zone edges (8°, 32°)
  is |C * sin(±12°)| ≈ 0.57 inches — nonzero but modest.

  This is a single smooth formula across all contact, which avoids the
  two-regime structure of the original (flat zero then growing penalty).

Physical constants
------------------
  C = (bat_diameter/2) + (ball_diameter/2) = 1.305 + 1.450 = 2.755 inches

Modifying this script
----------------------
FEATURES        pitch features for XGBoost fixed effects
CLUSTER_COL     random effects grouping ('batter' or 'batter_pitch_type')
XGB_PARAMS      XGBoost hyperparameters
MAX_ITER        MERF EM iterations
MIN_PA          minimum AB proxy for scoring output

Outputs
-------
  barrel_variants_v1_fitted.csv       fitted values, variant 1
  barrel_variants_v2_fitted.csv       fitted values, variant 2
  barrel_variants_v1_re.csv           random intercepts, variant 1
  barrel_variants_v2_re.csv           random intercepts, variant 2
  barrel_variants_v1_scores.csv       per-batter scores, variant 1
  barrel_variants_v2_scores.csv       per-batter scores, variant 2
  barrel_variants_v1_diagnostics.png  residual plots + GLL, variant 1
  barrel_variants_v2_diagnostics.png  residual plots + GLL, variant 2
  barrel_variants_v1_importance.png   feature importance, variant 1
  barrel_variants_v2_importance.png   feature importance, variant 2
  barrel_variants_comparison.csv      side-by-side score comparison

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

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS
CONTACT    = IN_PLAY | FOUL

# ── Physical constants ─────────────────────────────────────────────────────────
BAT_DIAMETER  = 2.61
BALL_DIAMETER = 2.90
C = (BAT_DIAMETER / 2) + (BALL_DIAMETER / 2)   # 2.755 inches

# ── Model configuration ────────────────────────────────────────────────────────
FEATURES = [
    'release_speed_c',
    'plate_x_bat_flip',
    'plate_z',
    # 'abs_pfx_x',    # uncomment to include
    # 'pfx_z',        # uncomment to include
    'intercept_y',
]

CLUSTER_COL = 'batter'

BATTER_AGG_FEATURES: list[str] = []

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

MAX_ITER = 15
MIN_PA   = 400

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'; DARK = '#1F2937'
VARIANT_COLORS = {'v1': '#7C3AED', 'v2': '#D97706'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. BARREL DISTANCE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_barrel_distance_v1(df: pd.DataFrame) -> pd.Series:
    """
    Variant 1: contact = 0, miss = miss_distance + C.

    All contact events are treated as perfect barrel placement regardless of
    launch angle. Only the miss/contact distinction matters.
    """
    bd = pd.Series(np.nan, index=df.index, dtype=float)

    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)

    # Misses: shift up by C to put on same scale as contact
    valid_miss = is_miss & df['miss_distance'].notna()
    bd.loc[valid_miss] = df.loc[valid_miss, 'miss_distance'] + C

    # All contact: zero regardless of launch angle
    bd.loc[is_contact] = 0.0

    return bd


def compute_barrel_distance_v2(df: pd.DataFrame) -> pd.Series:
    """
    Variant 2: 20°-centered single formula for contact, miss = miss_distance + C.

    Contact penalty: |C * sin(la - 20°)|
      la = 20°: sin(0)    = 0.000 → 0.00 inches  (perfect)
      la =  8°: sin(-12°) ≈ -0.208 → 0.57 inches
      la = 32°: sin( 12°) ≈  0.208 → 0.57 inches
      la =  0°: sin(-20°) ≈ -0.342 → 0.94 inches
      la = 60°: sin( 40°) ≈  0.643 → 1.77 inches
      la = 90°: sin( 70°) ≈  0.940 → 2.59 inches

    No flat reward region — 20° is the only zero. The original dual-boundary
    formula had a flat [8°, 32°] zero band; this version trades that for a
    single smooth curve centered on the ideal barrel angle.
    """
    bd = pd.Series(np.nan, index=df.index, dtype=float)

    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)

    # Misses: same as original
    valid_miss = is_miss & df['miss_distance'].notna()
    bd.loc[valid_miss] = df.loc[valid_miss, 'miss_distance'] + C

    # Contact: continuous penalty from 20°
    valid_contact = is_contact & df['launch_angle'].notna()
    la_rad = np.deg2rad(df.loc[valid_contact, 'launch_angle'])
    bd.loc[valid_contact] = np.abs(C * np.sin(la_rad - np.deg2rad(0)))

    # Contact with missing launch_angle: fall back to 0 (treat as neutral)
    # These are mostly foul tips where la wasn't recorded.
    # Change to np.nan to exclude them from the model instead.
    missing_la_contact = is_contact & df['launch_angle'].isna()
    bd.loc[missing_la_contact] = np.nan

    return bd


def print_variant_summary(bd: pd.Series, df: pd.DataFrame, label: str):
    """Print outcome distribution statistics for a barrel_distance variant."""
    is_miss    = df['description'].isin(MISS)
    is_contact = df['description'].isin(CONTACT)

    print(f'\n  [{label}] barrel_distance summary:')
    print(f'    overall:  mean={bd.mean():.3f}  median={bd.median():.3f}  '
          f'range=[{bd.min():.3f}, {bd.max():.3f}]')
    print(f'    misses:   mean={bd[is_miss].mean():.3f}  '
          f'n={is_miss.sum():,}')
    print(f'    contact:  mean={bd[is_contact].mean():.3f}  '
          f'zeros={( bd[is_contact] == 0).sum():,} / {is_contact.sum():,}  '
          f'({100*(bd[is_contact]==0).mean():.1f}% zero)')


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # ── Handedness corrections ─────────────────────────────────────────────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    df['abs_pfx_x'] = df['pfx_x'].abs()

    # ── Standardise release_speed ─────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── Rename intercept_y ────────────────────────────────────────────────────
    df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── String types ──────────────────────────────────────────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── AB proxy ──────────────────────────────────────────────────────────────
    bip_mask = df['description'].isin(IN_PLAY)
    k_mask   = df['events'] == 'strikeout'
    bip_ct   = df[bip_mask].groupby('batter').size().rename('n_bip')
    k_ct     = df[k_mask].groupby('batter').size().rename('n_k')
    ab_proxy = (pd.concat([bip_ct, k_ct], axis=1)
                  .fillna(0).astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])
                  [['ab_proxy']].reset_index())

    # ── Compute both barrel_distance variants in memory ───────────────────────
    df['barrel_distance_v1'] = compute_barrel_distance_v1(df)
    df['barrel_distance_v2'] = compute_barrel_distance_v2(df)

    print_variant_summary(df['barrel_distance_v1'].dropna(), df, 'v1')
    print_variant_summary(df['barrel_distance_v2'].dropna(), df, 'v2')

    # ── Filter to all swing events ────────────────────────────────────────────
    sub = df[df['description'].isin(ALL_SWINGS)].copy()
    print(f'\nAll swing events: {len(sub):,}  '
          f'(miss={sub["description"].isin(MISS).sum():,}  '
          f'foul={sub["description"].isin(FOUL).sum():,}  '
          f'in_play={sub["description"].isin(IN_PLAY).sum():,})')

    # ── Optional batter aggregate features ────────────────────────────────────
    if BATTER_AGG_FEATURES:
        agg = (sub.groupby('batter')[BATTER_AGG_FEATURES]
                  .mean().add_prefix('batter_mean_').reset_index())
        sub = sub.merge(agg, on='batter', how='left')

    # ── Attach AB proxy ───────────────────────────────────────────────────────
    sub = sub.merge(ab_proxy, on='batter', how='left')
    sub['ab_proxy'] = sub['ab_proxy'].fillna(0)

    # ── Grouping combo column ─────────────────────────────────────────────────
    sub['batter_pitch_type'] = sub['batter'] + '_' + sub['pitch_type'].fillna('UNK')

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)

    # ── Drop rows missing required features ───────────────────────────────────
    # Drop on features + cluster only; each variant drops separately on its outcome
    needed_base = [CLUSTER_COL] + FEATURES
    sub = sub.dropna(subset=needed_base).copy()
    print(f'After dropping NA on features: {len(sub):,} rows, '
          f'{sub[CLUSTER_COL].nunique():,} unique {CLUSTER_COL} groups')

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 3. MERF FITTING
# ══════════════════════════════════════════════════════════════════════════════

def build_inputs(sub: pd.DataFrame, outcome_col: str):
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols = FEATURES + pt_dummies

    if BATTER_AGG_FEATURES:
        X_cols += [f'batter_mean_{f}' for f in BATTER_AGG_FEATURES
                   if f'batter_mean_{f}' in sub.columns]

    # Drop rows where this variant's outcome is NaN
    mask = sub[outcome_col].notna()
    sub_v = sub[mask].copy()

    X        = sub_v[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub_v), 1))
    clusters = sub_v[CLUSTER_COL].reset_index(drop=True)
    y        = sub_v[outcome_col].reset_index(drop=True)

    return X, Z, clusters, y, X_cols, sub_v


def fit_merf(X, Z, clusters, y, label: str) -> MERF:
    xgb = XGBRegressor(**XGB_PARAMS)
    mrf = MERF(fixed_effects_model=xgb, max_iterations=MAX_ITER)

    print(f'\nFitting MERF [{label}]  '
          f'(n={len(y):,}, features={X.shape[1]}, max_iter={MAX_ITER})')
    print('GLL should increase and plateau at convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list,
                     out_prefix: str, label: str, color: str):

    resid = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Barrel Distance MERF – {label}',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.10, s=4, color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted'); ax.set_ylabel('Residual')
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
    fig.savefig(f'{out_prefix}_diagnostics.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out_prefix}_diagnostics.png')

    # Feature importance
    imp = pd.Series(mrf.trained_fe_model.feature_importances_,
                    index=X_cols).sort_values(ascending=True)
    n = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=color, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'XGBoost feature importance – {label}\n(top 25)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{out_prefix}_importance.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out_prefix}_importance.png')


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub_v: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series, outcome_col: str,
                   label: str) -> pd.DataFrame:

    ab_proxy_map = sub_v.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub_v[outcome_col].values - fitted)

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

    print(f'\n[{label}] Scores for {len(scores):,} batters (≥{MIN_PA} AB proxy):')
    print(scores[['batter', 'barrel_placement_score',
                  'mean_abs_resid', 'n_events', 'n_miss', 'n_contact']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════

def comparison_plot(scores_v1: pd.DataFrame, scores_v2: pd.DataFrame):
    """
    Scatter plot of v1 vs v2 barrel_placement_score for batters present in both.
    Useful for identifying where the two scoring systems disagree most.
    """
    merged = (scores_v1[['batter', 'barrel_placement_score']]
              .rename(columns={'barrel_placement_score': 'score_v1'})
              .merge(
                  scores_v2[['batter', 'barrel_placement_score']]
                  .rename(columns={'barrel_placement_score': 'score_v2'}),
                  on='batter', how='inner'))

    corr = np.corrcoef(merged['score_v1'], merged['score_v2'])[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(merged['score_v1'], merged['score_v2'],
               alpha=0.5, s=18, color=BLUE, rasterized=True)
    lim = [min(merged['score_v1'].min(), merged['score_v2'].min()),
           max(merged['score_v1'].max(), merged['score_v2'].max())]
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Barrel placement score – V1 (contact=0)')
    ax.set_ylabel('Barrel placement score – V2 (20° centered)')
    ax.set_title(f'V1 vs V2 scores  (r = {corr:.3f})\nn={len(merged):,} batters',
                 fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'barrel_variants_comparison.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')

    # Save merged comparison CSV
    merged.to_csv(os.path.join(OUT_DIR, 'barrel_variants_comparison.csv'),
                  index=False)

    return merged, corr


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Barrel Distance MERF – Variants\n')
    print(f'C = {C:.4f} inches  '
          f'(bat radius {BAT_DIAMETER/2:.4f} + ball radius {BALL_DIAMETER/2:.4f})')

    # ── Load and prep data ─────────────────────────────────────────────────────
    sub = prepare_data()

    all_scores = {}

    for label, outcome_col, color in [
        # ('v1 (contact=0)',       'barrel_distance_v1', VARIANT_COLORS['v1']),
        ('v2 (0° centered)',    'barrel_distance_v2', VARIANT_COLORS['v2']),
    ]:
        short = label.split()[0]   # 'v1' or 'v2'
        print(f'\n{"="*70}')
        print(f'VARIANT: {label}')
        print('='*70)

        # Build inputs (drops rows where this variant's outcome is NaN)
        X, Z, clusters, y, X_cols, sub_v = build_inputs(sub, outcome_col)

        # Fit
        mrf = fit_merf(X, Z, clusters, y, label)
        fitted = mrf.predict(X, Z, clusters)

        # R²
        ss_res = np.sum((y.values - fitted) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f'\nR²: {r2:.4f}')

        # Random intercepts
        re_df = (mrf.trained_b
                   .rename(columns={0: 'random_intercept'})
                   .reset_index()
                   .rename(columns={'index': CLUSTER_COL}))
        re_path = os.path.join(OUT_DIR, f'barrel_variants_{short}_re.csv')
        re_df.to_csv(re_path, index=False)
        print(f'Random intercepts → {re_path}')
        print('Best (b_i most negative):')
        print(re_df.sort_values('random_intercept').head(5).to_string(index=False))
        print('Worst (b_i most positive):')
        print(re_df.sort_values('random_intercept', ascending=False)
                   .head(5).to_string(index=False))

        # Fitted values
        fitted_df = pd.DataFrame({
            CLUSTER_COL:     clusters.values,
            'description':   sub_v['description'].values,
            outcome_col:     y.values,
            'fitted':        fitted,
            'residual':      y.values - fitted,
        })
        fitted_path = os.path.join(OUT_DIR, f'barrel_variants_{short}_fitted.csv')
        fitted_df.to_csv(fitted_path, index=False)
        print(f'Fitted values → {fitted_path}')

        # Diagnostics
        out_prefix = os.path.join(OUT_DIR, f'barrel_variants_{short}')
        diagnostic_plots(y, fitted, mrf, X_cols, out_prefix, label, color)

        # Scores
        scores = compute_scores(sub_v, fitted, clusters, outcome_col, label)
        scores_path = os.path.join(OUT_DIR, f'barrel_variants_{short}_scores.csv')
        scores.to_csv(scores_path, index=False)
        print(f'Scores → {scores_path}')

        all_scores[short] = scores

    # ── Cross-variant comparison ───────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('CROSS-VARIANT COMPARISON')
    print('='*70)
    merged, corr = comparison_plot(all_scores['v1'], all_scores['v2'])
    print(f'Score correlation V1 vs V2: {corr:.3f}  '
          f'(n={len(merged):,} batters in both)')

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()