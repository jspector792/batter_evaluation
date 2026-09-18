"""
int_y_merf.py
=============
Mixed Effects Random Forest (MERF) model predicting intercept_y (int_y) on
all swings, using XGBoost as the fixed-effects learner.

Background
----------
int_y is the y-position (depth) at which the ball crosses the batter's frontal
plane. It is a proxy for timing: a batter who makes contact further out in front
tends to be early; deeper contact tends to be late. The lmer model from the BPS
article was:

  int_y ~ 1 + (release_speed_c + plate_x_bat_flip | batter:pitch_type)

That model gave each batter×pitch_type group its own intercept AND its own
slopes on velocity and horizontal location (random slopes). This MERF version
translates that as follows:

  MERF fixed effects  (XGBoost):
      release_speed_c, plate_x_bat_flip, plate_z, pfx_x_bat_flip, pfx_z,
      pitch_type dummies
      → XGBoost learns the nonlinear, interactive population-level relationship
        between pitch characteristics and int_y, including the velocity and
        location effects that were fixed slopes in the lmer.

  MERF random effects  (GLS):
      One random intercept per batter (b_i ~ N(0, σ_b²))
      → captures each batter's persistent tendency to make contact deeper
        or shallower than pitch features predict.

  What is lost vs. lmer random slopes:
      The lmer allowed each batter to have their own *slope* on release_speed_c
      and plate_x_bat_flip (e.g. batter A is more affected by high velo than
      batter B). MERF cannot do this directly. Two approximations are available:

      Option A (BATTER_AGG_FEATURES, enabled by default):
          Append batter-level mean values of key features (mean bat speed,
          mean release_speed seen, etc.) to every row. XGBoost can then
          condition on batter characteristics and partially recover
          heterogeneous slopes.

      Option B (change CLUSTER_COL to 'batter_pitch_type'):
          Use batter×pitch_type as the grouping variable, giving a separate
          random intercept per cell — closer to the lmer nesting. Tradeoff:
          many more groups with less data each.

      Both options can be combined.

plate_x_bat_flip convention
----------------------------
plate_x is signed in the catcher's frame (positive = right side of plate).
Multiplying by -1 for RHB and +1 for LHB converts to a batter-relative frame
where positive always means "toward the outer edge of the plate". This matches
the timing model convention from swing_models_via_r.py.

pfx_x is similarly flipped (pfx_x_bat_flip) so horizontal break is always
expressed as toward/away from the batter regardless of pitcher handedness.

Modifying this model
---------------------
FEATURES        list below — add/remove pitch features freely.
                All numeric; pitch_type dummies are added automatically.
                To add intercept_x, simply append 'intercept_x' here and
                ensure INTERCEPT_X_COL is renamed in prepare_data().

CLUSTER_COL     grouping variable for random intercepts.
                'batter'            → one intercept per batter (default)
                'batter_pitch_type' → one per batter×pitch_type cell

BATTER_AGG_FEATURES
                Batter-level means appended to every row to help XGBoost
                approximate random slopes. Add column names from the raw
                data (e.g. 'bat_speed', 'launch_speed'). Set to [] to disable.

XGB_PARAMS      standard XGBoost knobs. Key ones:
                  max_depth        — tree depth (4-6 typical)
                  n_estimators     — number of trees
                  learning_rate    — smaller needs more trees but generalises
                  min_child_weight — raise to prevent overfitting on rare groups

MAX_ITER        MERF EM iterations. Watch GLL printout — flat = converged.

MIN_PA          minimum AB proxy for batter to appear in scoring output.

Outputs
-------
  int_y_merf_fitted.csv       fitted values + residuals per swing
  int_y_merf_re.csv           random intercept per batter (timing baseline)
  int_y_merf_scores.csv       per-batter timing scores (≥ MIN_PA)
  int_y_merf_diagnostics.png  residual plots + GLL convergence
  int_y_merf_importance.png   XGBoost feature importance

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw column names ───────────────────────────────────────────────────────────
INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

MISS      = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL      = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY   = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS = IN_PLAY | FOUL | MISS

# Platoon: batter handedness vs pitcher handedness
# same_hand = 1 when batter and pitcher throw/bat from same side
# Four-way platoon dummies give finer resolution than binary same/opposite
PLATOON_FEATURES = [
    'same_hand',       # binary: 1 = same side, 0 = opposite
    # 'platoon_RvR',   # uncomment to use four-way instead of binary
    # 'platoon_RvL',
    # 'platoon_LvR',
    # 'platoon_LvL',
]

# ── Model configuration ────────────────────────────────────────────────────────

# Fixed-effects features for XGBoost.
# pitch_type_* dummies are appended automatically — do not include pitch_type here.
# release_speed_c and plate_x_bat_flip are the core lmer fixed predictors;
# plate_z, pfx_x_bat_flip, pfx_z add pitch shape context that the lmer omitted.
FEATURES = [
    'release_speed_c',    # z-scored velocity — primary timing driver
    'plate_x_bat_flip',   # horizontal location, batter-relative (outer edge = +)
    'plate_z',            # vertical location
    'pfx_x_bat_flip',     # horizontal break, batter-relative (same flip as plate_x)
    'pfx_z',              # vertical break
    'same_hand',
]

# Random-effects grouping column.
# 'batter'            → scalar random intercept per batter (recommended default)
# 'batter_pitch_type' → random intercept per batter×pitch_type cell; more groups,
#                       less data per group, but closer to the lmer nesting
CLUSTER_COL = 'batter'

# Batter-level aggregate features merged onto every row.
# These let XGBoost approximate the per-batter random *slopes* from the lmer
# by conditioning on batter characteristics (e.g. a power hitter who typically
# sits back may respond differently to high velo than a contact hitter).
# Set to [] to disable. Values must be columns present in the raw parquet data.
BATTER_AGG_FEATURES: list[str] = ['bat_speed', 'release_speed']

# XGBoost hyperparameters.
XGB_PARAMS = dict(
    n_estimators       = 400,
    max_depth          = 5,
    learning_rate      = 0.04,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 20,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    random_state       = 42,
    n_jobs             = -1,
    verbosity          = 0,
)

MAX_ITER = 15    # EM iterations; increase to 25 if GLL hasn't plateaued
MIN_PA   = 400   # minimum AB proxy for scoring

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

    # ── String types first (needed for map lookups below) ─────────────────────
    for col in ['batter', 'pitch_type', 'stand']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── Handedness flip for plate_x ───────────────────────────────────────────
    # Converts catcher-frame plate_x to a batter-relative frame.
    # RHB: plate_x negated so positive = away from body (outer edge).
    # LHB: plate_x unchanged (already outer-positive in catcher frame).
    # This matches the timing model convention in swing_models_via_r.py.
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    print('plate_x_bat_flip applied (R→negated, L→unchanged)')

    # ── Handedness flip for pfx_x ─────────────────────────────────────────────
    # pfx_x is signed in the catcher frame (positive = pitcher's arm side).
    # Flipping for batter handedness means positive = break toward outer edge,
    # giving a consistent "moving away from barrel" interpretation.
    # This is conceptually similar to abs_pfx_x in the miss distance models
    # but preserves directionality (toward vs. away from batter).
    if 'pfx_x_bat_flip' not in df.columns:
        df['pfx_x_bat_flip'] = (
            df['pfx_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    print('pfx_x_bat_flip applied (R→negated, L→unchanged)')

    # ── Platoon features ──────────────────────────────────────────────────────
    # same_hand: 1 if batter and pitcher are on the same side, else 0.
    # This captures the systematic effect of arm-side vs. glove-side break
    # on contact depth that plate_x_bat_flip approximates but doesn't fully
    # isolate — a same-handed breaking ball moves differently relative to the
    # batter's eye than an opposite-handed one at the same plate location.
    if 'p_throws' in df.columns and 'stand' in df.columns:
        df['same_hand'] = (
            (df['stand'] == 'R') & (df['p_throws'] == 'R') |
            (df['stand'] == 'L') & (df['p_throws'] == 'L')
        ).astype(float)

        # Four-way platoon dummies (drop_first=False to keep all four;
        # XGBoost doesn't need a reference category). Uncomment PLATOON_FEATURES
        # above and swap 'same_hand' for the four dummies in FEATURES to use.
        df['platoon'] = df['stand'] + 'v' + df['p_throws']
        platoon_dummies = pd.get_dummies(df['platoon'], prefix='platoon')
        df = pd.concat([df, platoon_dummies], axis=1)
        print(f'Platoon distribution:\n'
              f'{df["platoon"].value_counts().to_string()}')
        print(f'same_hand rate: {df["same_hand"].mean():.3f}')
    else:
        print('WARNING: p_throws or stand missing — platoon features skipped')
        df['same_hand'] = 0.0

    # ── Standardise release_speed ─────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    # ── Rename int_y ──────────────────────────────────────────────────────────
    df = df.rename(columns={INTERCEPT_Y_COL: 'int_y'})

    # ── AB proxy from full dataset ────────────────────────────────────────────
    bip_mask = df['description'].isin(IN_PLAY)
    k_mask   = df['events'] == 'strikeout'
    bip_ct   = df[bip_mask].groupby('batter').size().rename('n_bip')
    k_ct     = df[k_mask].groupby('batter').size().rename('n_k')
    ab_proxy = (pd.concat([bip_ct, k_ct], axis=1)
                  .fillna(0).astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])
                  [['ab_proxy']].reset_index())

    # ── Filter to all swings ──────────────────────────────────────────────────
    # int_y is meaningful on any swing (in-play, foul, miss) because it
    # represents where the bat crossed the frontal plane regardless of outcome.
    sub = df[df['description'].isin(ALL_SWINGS)].copy()
    print(f'All swings: {len(sub):,}')

    # ── Batter-level aggregate features ──────────────────────────────────────
    # Computed on the swing subset so they reflect each batter's observed
    # characteristics during actual swing events, not all pitches seen.
    # These help XGBoost approximate the per-batter slopes from the lmer model.
    if BATTER_AGG_FEATURES:
        available = [f for f in BATTER_AGG_FEATURES if f in sub.columns]
        if available:
            agg = (sub.groupby('batter')[available]
                      .mean()
                      .add_prefix('batter_mean_')
                      .reset_index())
            sub = sub.merge(agg, on='batter', how='left')
            print(f'Batter aggregate features added: '
                  f'{[f"batter_mean_{f}" for f in available]}')
        else:
            print(f'Warning: none of BATTER_AGG_FEATURES found in data: '
                  f'{BATTER_AGG_FEATURES}')

    # ── Attach AB proxy ───────────────────────────────────────────────────────
    sub = sub.merge(ab_proxy, on='batter', how='left')
    sub['ab_proxy'] = sub['ab_proxy'].fillna(0)

    # ── batter_pitch_type combo column ────────────────────────────────────────
    # Available as CLUSTER_COL if you switch from 'batter' to 'batter_pitch_type'
    sub['batter_pitch_type'] = sub['batter'] + '_' + sub['pitch_type'].fillna('UNK')

    # ── One-hot encode pitch_type ──────────────────────────────────────────────
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)

    # ── Drop rows missing required columns ────────────────────────────────────
    needed = ['int_y', CLUSTER_COL] + FEATURES
    sub = sub.dropna(subset=needed).copy()
    print(f'After dropping NA: {len(sub):,} rows, '
          f'{sub[CLUSTER_COL].nunique():,} unique {CLUSTER_COL} groups')

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MERF INPUTS
# ══════════════════════════════════════════════════════════════════════════════

def build_merf_inputs(sub: pd.DataFrame):
    """
    MERF requires:
      X        — fixed-effects matrix (n × p): pitch features + pitch_type dummies
      Z        — random-effects matrix (n × q): column of ones for random intercept.
                 To add a random slope on e.g. release_speed_c, append that column:
                   Z = np.column_stack([np.ones(len(sub)),
                                        sub['release_speed_c'].values])
                 This increases model complexity and fitting time substantially.
      clusters — Series of cluster labels (n,)
      y        — outcome Series (n,)
    """
    pt_dummies = [c for c in sub.columns if c.startswith('pt_')]
    X_cols = FEATURES + pt_dummies

    # Add batter aggregate columns if they were created
    if BATTER_AGG_FEATURES:
        agg_cols = [f'batter_mean_{f}' for f in BATTER_AGG_FEATURES
                    if f'batter_mean_{f}' in sub.columns]
        X_cols += agg_cols

    X        = sub[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub), 1))           # random intercept only
    clusters = sub[CLUSTER_COL].reset_index(drop=True)
    y        = sub['int_y'].reset_index(drop=True)

    # Verify platoon feature survived the dropna
    for f in FEATURES:
        assert f in sub.columns, f'Feature {f!r} missing from modelling subset'

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
      learner = LGBMRegressor(n_estimators=400, ...)

    The GLL (generalised log-likelihood) printed each iteration should
    increase monotonically and plateau at convergence. If it is still
    climbing steeply at MAX_ITER, increase MAX_ITER.
    """
    xgb = XGBRegressor(**XGB_PARAMS)

    mrf = MERF(
        fixed_effects_model = xgb,
        max_iterations      = MAX_ITER,
    )

    print(f'\nFitting MERF  (cluster={CLUSTER_COL}, '
          f'features={X.shape[1]}, n={len(y):,}, max_iter={MAX_ITER})')
    print('GLL should increase and then plateau — that signals convergence.\n')

    mrf.fit(X, Z, clusters, y)
    return mrf


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS & PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y: pd.Series, fitted: np.ndarray,
                     mrf: MERF, X_cols: list, out_prefix: str):

    resid = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('int_y – MERF (XGBoost fixed effects)',
                 fontsize=13, fontweight='bold')

    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.08, s=3, color=BLUE, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted int_y'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    # Predicted vs actual
    ax = axes[0, 1]
    lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    ax.scatter(y, fitted, alpha=0.08, s=3, color=BLUE, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(y, fitted)[0, 1]
    ax.set_xlabel('Actual int_y'); ax.set_ylabel('Predicted int_y')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # GLL convergence trace
    ax = axes[1, 0]
    gll = mrf.gll_history
    ax.plot(range(1, len(gll) + 1), gll,
            color=GREEN, linewidth=2, marker='o', markersize=5)
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
    path = f'{out_prefix}_diagnostics_v2.png'
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
    path = f'{out_prefix}_importance_v2.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub: pd.DataFrame, fitted: np.ndarray,
                   clusters: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two complementary scoring outputs:

    1. timing_score  =  1 / mean(|residual|)  for batters with ab_proxy >= MIN_PA
       Measures within-batter timing *consistency* around the model prediction.
       Higher = smaller average unexplained deviation = more consistent timing.

    2. random_intercept  (from mrf.trained_b)
       Measures each batter's persistent baseline shift in int_y after
       controlling for pitch features. Interpretation:
         b_i > 0 → batter makes contact deeper (later) than pitch features predict
         b_i < 0 → batter makes contact further out front (earlier) than predicted
       Neither direction is inherently better — it reflects timing tendencies.
       The *magnitude* of b_i relative to other batters is the skill signal.

    These are saved separately; downstream you can combine them as needed.
    """
    ab_proxy_map = sub.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying   = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    resid_abs = np.abs(sub['int_y'].values - fitted)

    scores = (
        pd.DataFrame({
            'batter':    clusters.values,
            'abs_resid': resid_abs,
            'ab_proxy':  sub['ab_proxy'].values,
        })
        .groupby('batter')
        .agg(
            n_swings       = ('abs_resid', 'size'),
            mean_abs_resid = ('abs_resid', 'mean'),
            ab_proxy       = ('ab_proxy', 'first'),
        )
        .loc[lambda df: df.index.isin(qualifying)]
        .assign(timing_score=lambda df: 1.0 / df['mean_abs_resid'])
        .sort_values('timing_score', ascending=False)
        .reset_index()
    )

    print(f'\nTiming scores for {len(scores):,} batters (≥{MIN_PA} AB proxy):')
    print(scores[['batter', 'timing_score', 'mean_abs_resid', 'n_swings']]
          .head(10).to_string(index=False))

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('int_y Timing – MERF (XGBoost fixed effects)\n')

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
    # b_i < 0: contact further out front (early tendency)
    # b_i > 0: contact deeper (late tendency)
    re_df = (mrf.trained_b
               .rename(columns={0: 'random_intercept'})
               .reset_index()
               .rename(columns={'index': CLUSTER_COL}))
    re_path = os.path.join(OUT_DIR, 'int_y_merf_re_v2.csv')
    re_df.to_csv(re_path, index=False)
    print(f'\nRandom intercepts (timing baseline) → {re_path}')
    print('Most out-front batters (b_i most negative):')
    print(re_df.sort_values('random_intercept').head(5).to_string(index=False))
    print('Latest batters (b_i most positive):')
    print(re_df.sort_values('random_intercept', ascending=False)
               .head(5).to_string(index=False))

    # ── Fitted values CSV ─────────────────────────────────────────────────────
    fitted_df = pd.DataFrame({
        CLUSTER_COL: clusters.values,
        'int_y':     y.values,
        'fitted':    fitted,
        'residual':  y.values - fitted,
    })
    fitted_path = os.path.join(OUT_DIR, 'int_y_merf_fitted_v2.csv')
    fitted_df.to_csv(fitted_path, index=False)
    print(f'Fitted values → {fitted_path}')

    # ── Diagnostics ───────────────────────────────────────────────────────────
    out_prefix = os.path.join(OUT_DIR, 'int_y_merf')
    diagnostic_plots(y, fitted, mrf, X_cols, out_prefix)

    # ── Scores ────────────────────────────────────────────────────────────────
    scores = compute_scores(sub, fitted, clusters)
    scores_path = os.path.join(OUT_DIR, 'int_y_merf_scores_v2.csv')
    scores.to_csv(scores_path, index=False)
    print(f'Timing scores → {scores_path}')

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()