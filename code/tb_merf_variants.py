"""
tb_merf_variants.py
===================
Total-bases analog of contact_merf_variants.py (merf_variants_v3.py).

Same architecture -- MERF (XGBoost fixed effects + batter random intercept),
same predictor blocks, same diagnostics/scoring/comparison outputs -- but the
outcome is TOTAL BASES GAINED ON THE SWING instead of launch angle / signed
barrel distance / launch speed.

  total_bases = single 1, double 2, triple 3, home_run 4, everything else 0.

Why this is a different question from the V1/V2/V3 outcomes
-----------------------------------------------------------
V1-V3 all model an intermediate, *geometric* quality of contact (where on the
bat, at what angle, how hard) and score a batter by how CONSISTENTLY they hit
their expected value -- 1/mean|residual|. Total bases is the terminal payoff,
so consistency is the wrong scoring direction: you want to systematically
EXCEED the pitch-context expectation, not sit on it. The scoring block here is
adapted accordingly (see compute_scores) -- signed residual against the
fixed-effect-only prediction, which is the direct analog of
rb_contact/step6's "Contact Skill Added".

Variants
--------
tb1 -- all contact events (fouls + balls in play), fouls scored 0 TB
       Features: V1V2_FEATURES (release_speed_c, plate_x_bat_flip, plate_z,
                 intercept_y, attack_angle, swing_path_tilt) + pitch_type
       Reads as "damage per swing that touched the ball". Fouls are a real
       outcome of a swing decision, so folding them in as zeros is a
       defensible population -- but it makes the outcome ~85% zeros.

tb2 -- balls in play only
       Features: same as tb1
       Drops the foul zero-mass; reads as "damage per batted ball".

tb3 -- balls in play only, + bat_speed_c
       Features: tb2's + bat_speed_c
       Ablation mirroring V1-vs-V3 in the source script: how much does bat
       speed add once pitch location/timing/swing geometry are controlled for.
       Kept as a superset (release_speed_c retained) rather than a swap, so
       the tb2 -> tb3 delta is a clean marginal-value read on bat speed.

       RESULT: do not use this variant for batter scoring. It has the best
       held-out R^2 of the three (0.066 vs 0.059) and the worst batter score
       by a mile (split-half rho 0.20 vs 0.49). bat_speed is a persistent
       batter trait, so XGBoost puts it in the FIXED effect -- the "pitch
       context" expectation ends up correlating 0.75 with the batter's own
       mean bat speed, and tb_above_expected then subtracts the batter's
       power out of their own score. Prediction accuracy and batter-metric
       validity point in opposite directions here; see context_contamination().

Distributional caveat (stated up front, checked in the diagnostics)
-------------------------------------------------------------------
MERF's EM assumes Gaussian errors. Total bases is a bounded, discrete,
heavily zero-inflated count (85% zeros on contact, 65% on balls in play), so
the residuals will NOT be Gaussian and the residual histograms will show it.
The point estimates (conditional mean TB) are still the right thing -- least
squares gives an unbiased conditional-mean fit regardless -- but the GLL
values and any variance-based inference from the EM are not trustworthy as
likelihoods. Judge these models on held-out R^2 and on the reliability of the
batter score, both reported below, not on GLL.

Outputs (out/tb_models/)
------------------------
  tb{1,2,3}_fitted.csv        per-swing outcome / fitted / fe-only / residual
  tb{1,2,3}_re.csv            batter random intercepts
  tb{1,2,3}_scores.csv        per-batter scores
  tb{1,2,3}_diagnostics.png   residual plots + GLL
  tb{1,2,3}_importance.png    feature importance
  tb{1,2,3}_model.joblib      saved model
  tb_comparison.csv           R^2 (in-sample + held-out), reliability table
  tb_evaluation_report.md     written summary

Re-running
----------
  python code/tb_merf_variants.py                  full run (~50 min, 6 MERF fits)
  python code/tb_merf_variants.py --reuse-models   rescore/regenerate reports from
                                                   the saved .joblib fits (seconds)
  python code/tb_merf_variants.py --sample-n 25000 --min-pa 50    smoke test

Requires: merf, xgboost, joblib, scikit-learn, pandas, numpy, matplotlib, seaborn
"""

import os, glob, argparse, warnings
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'all_pitches_2025')
OUT_DIR  = os.path.join(BASE_DIR, 'out', 'tb_models')
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

FOUL    = {'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip'}
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
CONTACT = IN_PLAY | FOUL

# Total bases by terminal event. Anything not listed (field_out, force_out,
# GIDP, sac_fly, field_error, fielders_choice, catcher_interf, strikeout on a
# caught foul tip, or a foul with no terminal event at all) scores 0.
# field_error is deliberately 0: the batter reached, but no base was *earned*,
# which is the standard total-bases convention.
TB_MAP = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}

V1V2_FEATURES = [
    'release_speed_c',
    'plate_x_bat_flip',
    'plate_z',
    'intercept_y',
    'attack_angle',
    'swing_path_tilt',
]
TB3_FEATURES = V1V2_FEATURES + ['bat_speed_c']

CLUSTER_COL = 'batter'
MIN_PA      = 400    # ab_proxy threshold for the batter score table (same as source)
BAT_SPEED_MIN = 50.0  # filters bunts / check swings (same as source)

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
# Same rationale as rb_contact/oof_timing_offset.py: without this the EM grinds
# to the iteration ceiling long after GLL has plateaued.
GLL_EARLY_STOP = 1e-4

SEED = 42
HOLDOUT_FRAC = 0.20

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'
VARIANT_COLORS = {'tb1': '#7C3AED', 'tb2': '#D97706', 'tb3': '#0891B2'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    for col in ['batter', 'pitch_type', 'stand']:
        df[col] = df[col].astype(str)

    df['plate_x_bat_flip'] = df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)

    # z-scored on the full pitch population, before any filtering (same as source)
    rs_mean, rs_std = df['release_speed'].mean(), df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed_c: mean={rs_mean:.2f}, sd={rs_std:.2f}')

    df = df.rename(columns={INTERCEPT_Y_COL: 'intercept_y'})

    # ── Total bases ───────────────────────────────────────────────────────────
    df['total_bases'] = df['events'].map(TB_MAP).fillna(0.0)

    # ── AB proxy (balls in play + strikeouts), same as source ─────────────────
    bip_ct = df[df['description'].isin(IN_PLAY)].groupby('batter').size().rename('n_bip')
    k_ct   = df[df['events'] == 'strikeout'].groupby('batter').size().rename('n_k')
    ab_proxy = (pd.concat([bip_ct, k_ct], axis=1).fillna(0).astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])[['ab_proxy']]
                  .reset_index())
    df = df.merge(ab_proxy, on='batter', how='left')
    df['ab_proxy'] = df['ab_proxy'].fillna(0)

    sub = df[df['description'].isin(CONTACT)].copy()
    print(f'Contact events: {len(sub):,}  '
          f'(in_play={sub["description"].isin(IN_PLAY).sum():,}  '
          f'foul={sub["description"].isin(FOUL).sum():,})')

    n_before = len(sub)
    sub = sub[sub['bat_speed'] >= BAT_SPEED_MIN].copy()
    print(f'After bat_speed >= {BAT_SPEED_MIN} filter: {len(sub):,} '
          f'(removed {n_before - len(sub):,})')

    bs_mean, bs_std = sub['bat_speed'].mean(), sub['bat_speed'].std()
    sub['bat_speed_c'] = (sub['bat_speed'] - bs_mean) / bs_std
    print(f'bat_speed_c: mean={bs_mean:.2f}, sd={bs_std:.2f}')

    sub['is_bip'] = sub['description'].isin(IN_PLAY)
    sub = pd.get_dummies(sub, columns=['pitch_type'], prefix='pt', drop_first=False)

    print(f'Unique batters: {sub[CLUSTER_COL].nunique():,}')
    print(f'total_bases (all contact): mean={sub["total_bases"].mean():.4f} '
          f'sd={sub["total_bases"].std():.4f} zero_frac={(sub["total_bases"] == 0).mean():.4f}')
    bip = sub[sub['is_bip']]
    print(f'total_bases (BIP only):    mean={bip["total_bases"].mean():.4f} '
          f'sd={bip["total_bases"].std():.4f} zero_frac={(bip["total_bases"] == 0).mean():.4f}')
    return sub


def build_inputs(sub: pd.DataFrame, features: list, outcome_col: str) -> tuple:
    """Drop rows missing the outcome or any feature; return MERF inputs."""
    pt_dummies = sorted(c for c in sub.columns if c.startswith('pt_'))
    X_cols = features + pt_dummies

    needed = [outcome_col, CLUSTER_COL] + features
    n_before = len(sub)
    sub_v = sub.dropna(subset=needed).copy()
    print(f'  After dropping NA: {len(sub_v):,} rows (dropped {n_before - len(sub_v):,})')

    X        = sub_v[X_cols].astype(float).reset_index(drop=True)
    Z        = np.ones((len(sub_v), 1))
    clusters = sub_v[CLUSTER_COL].reset_index(drop=True)
    y        = sub_v[outcome_col].reset_index(drop=True)
    return X, Z, clusters, y, X_cols, sub_v


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT
# ══════════════════════════════════════════════════════════════════════════════

def fit_merf(X, Z, clusters, y, label: str) -> MERF:
    xgb = XGBRegressor(**XGB_PARAMS)
    mrf = MERF(fixed_effects_model=xgb, max_iterations=MAX_ITER,
               gll_early_stop_threshold=GLL_EARLY_STOP)
    print(f'\n  Fitting MERF [{label}]  (n={len(y):,}, features={X.shape[1]}, '
          f'clusters={clusters.nunique():,}, max_iter={MAX_ITER})')
    mrf.fit(X, Z, clusters, y)
    return mrf


def r2(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def holdout_r2(X, Z, clusters, y, label: str) -> dict:
    """
    Honest out-of-sample read. A MERF fit on all rows and scored on those same
    rows is optimistic twice over: XGBoost has seen every row, and each batter's
    random intercept was estimated from the very rows it's being scored on --
    with ~500 batters and ~250k swings that second effect is small but it is not
    zero, and it is exactly the effect the batter score depends on.

    Row-level 80/20 split (matching the rb_contact pipeline's swing-level fold
    discipline), then three held-out R^2 values:
      full     -- fixed effect + that batter's random intercept
      fe_only  -- fixed effect alone (context-only prediction)
    The gap between them is the random intercept's genuine out-of-sample value.
    """
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(len(y)) < HOLDOUT_FRAC
    tr, te = ~test_mask, test_mask
    print(f'  Holdout split: train={tr.sum():,} test={te.sum():,}')

    mrf = fit_merf(X[tr].reset_index(drop=True), Z[tr],
                   clusters[tr].reset_index(drop=True),
                   y[tr].reset_index(drop=True), f'{label} [holdout-train]')

    X_te, cl_te = X[te].reset_index(drop=True), clusters[te].reset_index(drop=True)
    y_te = y[te].reset_index(drop=True)
    pred_full = mrf.predict(X_te, Z[te], cl_te)
    pred_fe   = mrf.trained_fe_model.predict(X_te)
    return dict(holdout_r2_full=r2(y_te, pred_full),
                holdout_r2_fe_only=r2(y_te, pred_fe),
                holdout_n_test=int(te.sum()))


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_plots(y, fitted, mrf, X_cols, short, display_label, outcome_col):
    color = VARIANT_COLORS[short]
    resid = y.values - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Total-Bases MERF - {display_label}', fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.08, s=4, color=color, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel(f'Fitted {outcome_col}'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold'); ax.grid(alpha=0.3)

    # Actual TB is discrete {0,1,2,3,4}, so a raw scatter of actual-vs-predicted
    # is five horizontal stripes and reads as nothing. Mean fitted value within
    # each actual-TB level is the informative version of the same plot.
    ax = axes[0, 1]
    levels = sorted(pd.Series(y.values).unique())
    means = [fitted[y.values == lv].mean() for lv in levels]
    counts = [(y.values == lv).sum() for lv in levels]
    ax.bar([str(int(lv)) for lv in levels], means, color=color, alpha=0.85, edgecolor='white')
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, m, f'n={c:,}', ha='center', va='bottom', fontsize=8)
    corr = float(np.corrcoef(y.values, fitted)[0, 1])
    ax.set_xlabel('Actual total bases'); ax.set_ylabel('Mean fitted')
    ax.set_title(f'Mean fitted by actual TB  (r = {corr:.3f})', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    gll = mrf.gll_history
    ax.plot(range(1, len(gll) + 1), gll, color=GREEN, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('EM Iteration'); ax.set_ylabel('GLL')
    ax.set_title('MERF convergence\n(Gaussian GLL on a zero-inflated count -- '
                 'convergence diagnostic only)', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lo, hi = np.percentile(resid, 0.5), np.percentile(resid, 99.5)
    ax.hist(np.clip(resid, lo, hi), bins=80, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (sigma = {resid.std():.3f}, skew = {stats.skew(resid):.2f})',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{short}_diagnostics.png')
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
    print(f'  -> {path}')

    imp = pd.Series(mrf.trained_fe_model.feature_importances_,
                    index=X_cols).sort_values(ascending=True)
    n = min(25, len(imp))
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    imp.tail(n).plot.barh(ax=ax, color=color, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title(f'XGBoost feature importance - {display_label}\n(top 25)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{short}_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
    print(f'  -> {path}')
    return corr


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub_v, y, fitted, fe_only, clusters, mrf, short) -> pd.DataFrame:
    """
    Adapted from the source script's 1/mean|residual|, which does not transfer.

    For launch angle / barrel distance, small |residual| = good: you want to
    land on the expected value. For total bases you want to BEAT it, and the
    signed residual against the FIXED-EFFECT-ONLY prediction is the quantity
    that says so:

      tb_above_expected = mean( total_bases - fe_only_prediction )

    fe_only is the pitch-context expectation with the batter's own random
    intercept held out of the prediction -- the same "what would a league-average
    batter do against these exact pitches" construction as rb_contact/step6's
    Contact Skill Added, obtained here directly from the fitted fixed-effects
    model rather than via a placeholder cluster ID.

    Reported alongside:
      random_intercept -- MERF's shrunk b_i, the model's own version of the
                          same quantity (shrunk toward 0 by sample size)
      mean_abs_resid   -- the source script's consistency term, kept for
                          continuity; here it mostly tracks a batter's power
                          (big hits are big residuals), so it is NOT a skill
                          score and is not inverted into one.
    """
    ab_proxy_map = sub_v.groupby(CLUSTER_COL)['ab_proxy'].first()
    qualifying = ab_proxy_map[ab_proxy_map >= MIN_PA].index

    per_swing = pd.DataFrame({
        'batter':    clusters.values,
        'tb':        y.values,
        'fitted':    fitted,
        'fe_only':   fe_only,
        'abs_resid': np.abs(y.values - fitted),
        'ab_proxy':  sub_v['ab_proxy'].values,
        'bat_speed': sub_v['bat_speed'].values,
    })
    per_swing['above_expected'] = per_swing['tb'] - per_swing['fe_only']

    scores = (per_swing.groupby('batter')
              .agg(n_contact=('tb', 'size'),
                   mean_tb=('tb', 'mean'),
                   expected_tb=('fe_only', 'mean'),
                   tb_above_expected=('above_expected', 'mean'),
                   mean_abs_resid=('abs_resid', 'mean'),
                   mean_bat_speed=('bat_speed', 'mean'),
                   ab_proxy=('ab_proxy', 'first'))
              .loc[lambda d: d.index.isin(qualifying)]
              .sort_values('tb_above_expected', ascending=False)
              .reset_index())

    re_df = (mrf.trained_b.rename(columns={0: 'random_intercept'})
             .reset_index().rename(columns={'index': CLUSTER_COL}))
    re_df[CLUSTER_COL] = re_df[CLUSTER_COL].astype(str)
    scores = scores.merge(re_df, on=CLUSTER_COL, how='left')

    print(f'\n  Scores for {len(scores):,} batters (>={MIN_PA} AB proxy):')
    print('  Top 10 by TB above expected:')
    print(scores[['batter', 'mean_tb', 'expected_tb', 'tb_above_expected',
                  'random_intercept', 'n_contact']].head(10).to_string(index=False))
    print('  Bottom 5:')
    print(scores[['batter', 'mean_tb', 'expected_tb', 'tb_above_expected',
                  'random_intercept', 'n_contact']].tail(5).to_string(index=False))
    if scores['random_intercept'].notna().sum() > 2:
        rho = stats.spearmanr(scores['tb_above_expected'],
                              scores['random_intercept']).correlation
        print(f'  Spearman(tb_above_expected, random_intercept) = {rho:.3f}')

    contam = context_contamination(scores)
    print(f'  Context contamination: corr(expected_tb, batter mean bat speed) = '
          f'{contam:.3f}  {"<-- WARNING, see docstring" if abs(contam) > 0.4 else ""}')
    return scores, per_swing


def context_contamination(scores: pd.DataFrame) -> float:
    """
    Does the "pitch context" expectation secretly encode the batter?

    fe_only is supposed to answer "what would a league-average batter do
    against these exact pitches", so a batter's mean expected_tb should
    reflect the pitches they faced, not who they are. If a persistent
    batter-level trait is in the feature set, XGBoost puts that trait into
    the fixed effect, expected_tb starts tracking the batter, and
    tb_above_expected subtracts away the very skill it is meant to measure.

    Batter mean bat speed is the cleanest probe: it is close to a pure
    identity trait. |corr| > ~0.4 means the context model is contaminated
    and the resulting batter score should not be trusted -- this is exactly
    what separates TB3 (bat_speed_c in the features) from TB1/TB2.
    """
    if len(scores) < 3 or scores['mean_bat_speed'].isna().all():
        return float('nan')
    return float(scores['mean_bat_speed'].corr(scores['expected_tb']))


# ══════════════════════════════════════════════════════════════════════════════
# 5. RELIABILITY (does the batter score measure a real, persistent skill?)
# ══════════════════════════════════════════════════════════════════════════════

def discrimination(group_means: pd.Series, within_var: pd.Series, n: pd.Series) -> float:
    """
    Franks et al. signal/noise decomposition, general (non-binomial) form --
    rb_contact/step5 uses the binomial special case p(1-p)/n, which does not
    apply to total bases. Here the sampling variance of each batter's mean is
    estimated directly as that batter's own within-batter variance / n.

      discrimination = max(0, var(means) - mean(sampling var)) / var(means)

    Fraction of the observed between-batter spread that is real skill rather
    than sampling noise. Higher is better; 0 means the spread is pure noise.
    """
    total_var = group_means.var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return float('nan')
    return float(max(0.0, total_var - (within_var / n).mean()) / total_var)


def reliability(per_swing: pd.DataFrame, game_date: pd.Series, min_n: int = 100) -> dict:
    """
    Split each batter's contact events into first/second half of the season by
    date and correlate the two halves. A metric that reflects a persistent
    batter skill should reproduce across halves; one that is mostly luck on
    batted-ball outcomes should not.

    Compares tb_above_expected (model-based) against raw mean_tb (the naive
    metric it has to beat to be worth anything).
    """
    ps = per_swing.assign(game_date=pd.to_datetime(game_date.values))
    median_date = ps['game_date'].median()
    halves = {'h1': ps[ps['game_date'] <= median_date], 'h2': ps[ps['game_date'] > median_date]}

    agg = {}
    for k, h in halves.items():
        g = h.groupby('batter')
        agg[k] = pd.DataFrame({
            'n': g.size(),
            'tb_above_expected': g['above_expected'].mean(),
            'mean_tb': g['tb'].mean(),
        }).query(f'n >= {min_n}')

    m = agg['h1'].join(agg['h2'], how='inner', lsuffix='_h1', rsuffix='_h2')

    out = {'n_batters': int(len(m))}
    for metric in ['tb_above_expected', 'mean_tb']:
        if len(m) > 2:
            rho, p = stats.spearmanr(m[f'{metric}_h1'], m[f'{metric}_h2'])
            r = float(np.corrcoef(m[f'{metric}_h1'], m[f'{metric}_h2'])[0, 1])
        else:
            rho, p, r = np.nan, np.nan, np.nan
        out[metric] = dict(spearman_rho=float(rho), spearman_p=float(p), pearson_r=r)

    # Full-season discrimination for both metrics
    g = per_swing.groupby('batter')
    full = pd.DataFrame({'n': g.size(),
                         'tb_above_expected': g['above_expected'].mean(),
                         'mean_tb': g['tb'].mean(),
                         'var_above': g['above_expected'].var(ddof=1),
                         'var_tb': g['tb'].var(ddof=1)}).query(f'n >= {min_n}')
    out['tb_above_expected']['discrimination'] = discrimination(
        full['tb_above_expected'], full['var_above'], full['n'])
    out['mean_tb']['discrimination'] = discrimination(full['mean_tb'], full['var_tb'], full['n'])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    dict(short='tb1', display_label='TB1: total bases, all contact (fouls = 0)',
         population='contact', features=V1V2_FEATURES),
    dict(short='tb2', display_label='TB2: total bases, balls in play only',
         population='bip', features=V1V2_FEATURES),
    dict(short='tb3', display_label='TB3: total bases, balls in play + bat speed',
         population='bip', features=TB3_FEATURES),
]


def main():
    global MIN_PA
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-n', type=int, default=None,
                        help='Subsample N contact events for a smoke test')
    parser.add_argument('--variants', default='tb1,tb2,tb3',
                        help='Comma-separated variant shorts to run')
    parser.add_argument('--skip-holdout', action='store_true',
                        help='Skip the 80/20 held-out refit (halves runtime, '
                             'but leaves only optimistic in-sample R^2)')
    parser.add_argument('--min-pa', type=int, default=MIN_PA)
    parser.add_argument('--reuse-models', action='store_true',
                        help='Load {short}_model.joblib instead of refitting, and '
                             'carry the holdout_* columns over from the existing '
                             'tb_comparison.csv. For regenerating scores/reports '
                             'after a scoring change without paying for the EM again.')
    args = parser.parse_args()
    MIN_PA = args.min_pa

    cached_holdout = {}
    if args.reuse_models:
        comp_path = os.path.join(OUT_DIR, 'tb_comparison.csv')
        if os.path.exists(comp_path):
            prev = pd.read_csv(comp_path).set_index('short')
            ho_cols = [c for c in prev.columns if c.startswith('holdout_')]
            cached_holdout = prev[ho_cols].to_dict('index')

    print('Total-Bases MERF Variants\n')
    sub = prepare_data()
    if args.sample_n and args.sample_n < len(sub):
        sub = sub.sample(n=args.sample_n, random_state=SEED).reset_index(drop=True)
        print(f'Subsampled to {len(sub):,} rows for smoke test')

    wanted = [v.strip() for v in args.variants.split(',')]
    records, all_scores, reliab = [], {}, {}

    for cfg in VARIANTS:
        if cfg['short'] not in wanted:
            continue
        short, label, features = cfg['short'], cfg['display_label'], cfg['features']
        print(f'\n{"="*70}\nVARIANT: {label}\nOutcome: total_bases\n{"="*70}')

        pop = sub if cfg['population'] == 'contact' else sub[sub['is_bip']].copy()
        print(f'  Population: {cfg["population"]} ({len(pop):,} rows)')

        X, Z, clusters, y, X_cols, sub_v = build_inputs(pop, features, 'total_bases')

        model_path = os.path.join(OUT_DIR, f'{short}_model.joblib')
        if args.reuse_models and os.path.exists(model_path):
            print(f'  Reusing saved model {model_path} (no refit)')
            mrf = joblib.load(model_path)
            ho = cached_holdout.get(short, {})
        else:
            ho = {} if args.skip_holdout else holdout_r2(X, Z, clusters, y, label)
            mrf = fit_merf(X, Z, clusters, y, label)
            joblib.dump(mrf, model_path, compress=3)

        fitted  = mrf.predict(X, Z, clusters)
        fe_only = mrf.trained_fe_model.predict(X)

        r2_in      = r2(y, fitted)
        r2_in_fe   = r2(y, fe_only)
        print(f'\n  In-sample R^2 (full)    : {r2_in:.4f}')
        print(f'  In-sample R^2 (fe only) : {r2_in_fe:.4f}')
        for k, v in ho.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v:,}')

        re_df = (mrf.trained_b.rename(columns={0: 'random_intercept'})
                 .reset_index().rename(columns={'index': CLUSTER_COL}))
        re_df.to_csv(os.path.join(OUT_DIR, f'{short}_re.csv'), index=False)

        pd.DataFrame({
            CLUSTER_COL: clusters.values,
            'description': sub_v['description'].values,
            'events': sub_v['events'].values,
            'total_bases': y.values,
            'fitted': fitted,
            'fe_only': fe_only,
            'residual': y.values - fitted,
        }).to_csv(os.path.join(OUT_DIR, f'{short}_fitted.csv'), index=False)

        corr = diagnostic_plots(y, fitted, mrf, X_cols, short, label, 'total_bases')
        scores, per_swing = compute_scores(sub_v, y, fitted, fe_only, clusters, mrf, short)
        scores.to_csv(os.path.join(OUT_DIR, f'{short}_scores.csv'), index=False)

        rel = reliability(per_swing, sub_v['game_date'])
        reliab[short] = rel
        print(f'\n  Split-half reliability (n={rel["n_batters"]} batters):')
        for metric in ['tb_above_expected', 'mean_tb']:
            r = rel[metric]
            print(f'    {metric:20s} spearman={r["spearman_rho"]:.3f}  '
                  f'discrimination={r["discrimination"]:.3f}')

        all_scores[short] = scores
        records.append(dict(variant=label, short=short, outcome='total_bases',
                            population=cfg['population'], n_obs=len(y),
                            n_features=X.shape[1], n_batters=int(clusters.nunique()),
                            r2_in_sample=round(r2_in, 4),
                            r2_in_sample_fe_only=round(r2_in_fe, 4),
                            pred_vs_actual_r=round(corr, 4),
                            **{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in ho.items()},
                            reliab_rho_tb_above_expected=round(rel['tb_above_expected']['spearman_rho'], 4),
                            reliab_rho_mean_tb=round(rel['mean_tb']['spearman_rho'], 4),
                            discrim_tb_above_expected=round(rel['tb_above_expected']['discrimination'], 4),
                            discrim_mean_tb=round(rel['mean_tb']['discrimination'], 4),
                            context_contamination=round(context_contamination(scores), 4)))

    comp = pd.DataFrame(records)
    comp.to_csv(os.path.join(OUT_DIR, 'tb_comparison.csv'), index=False)
    print(f'\nVariant comparison:\n{comp.to_string(index=False)}')

    # Cross-variant score agreement
    shorts = list(all_scores)
    if len(shorts) > 1:
        print('\nBatter-score agreement (Spearman on tb_above_expected):')
        for i in range(len(shorts)):
            for j in range(i + 1, len(shorts)):
                a = all_scores[shorts[i]].set_index('batter')['tb_above_expected']
                b = all_scores[shorts[j]].set_index('batter')['tb_above_expected']
                joined = pd.concat([a, b], axis=1, join='inner')
                if len(joined) > 2:
                    rho = stats.spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).correlation
                    print(f'  {shorts[i]} vs {shorts[j]}: {rho:.3f}  (n={len(joined)})')

    write_report(comp, reliab, all_scores)
    print(f'\nAll outputs in {OUT_DIR}')


def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table -- pandas.to_markdown needs tabulate, which isn't
    in this env (same hand-rolled approach as rb_contact/step1_2)."""
    header = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    body = '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |'
                     for row in df.itertuples(index=False))
    return '\n'.join([header, sep, body])


def write_report(comp: pd.DataFrame, reliab: dict, all_scores: dict):
    lines = ['# Total-Bases MERF - Evaluation Report\n',
             'Outcome: total bases gained on the swing '
             '(single 1 / double 2 / triple 3 / HR 4, everything else 0).',
             'Architecture: MERF (XGBoost fixed effects + batter random intercept), '
             'matching contact_merf_variants.py.\n',
             '## Variant comparison\n', md_table(comp), '\n',
             '## Split-half reliability & discrimination\n',
             'Split-half: batter score computed separately on each half of the season '
             '(split at the median game date, >=100 contact events per half), '
             'Spearman across halves. Discrimination: fraction of between-batter '
             'variance that is skill rather than sampling noise.\n']
    rows = []
    for short, rel in reliab.items():
        for metric in ['tb_above_expected', 'mean_tb']:
            rows.append(dict(variant=short, metric=metric, n_batters=rel['n_batters'],
                             spearman_rho=round(rel[metric]['spearman_rho'], 4),
                             pearson_r=round(rel[metric]['pearson_r'], 4),
                             discrimination=round(rel[metric]['discrimination'], 4)))
    lines += [md_table(pd.DataFrame(rows)), '\n', '## Interpretation\n']

    # Auto-generated verdict per variant: the model-based batter score only
    # earns its keep if it is MORE reliable across halves than raw mean TB,
    # and only means what it claims if the context expectation isn't
    # contaminated by batter identity.
    for _, r in comp.iterrows():
        delta = r['reliab_rho_tb_above_expected'] - r['reliab_rho_mean_tb']
        verdict = ('beats raw mean TB' if delta > 0.02 else
                   'no better than raw mean TB' if delta > -0.02 else
                   'WORSE than raw mean TB')
        contam = r.get('context_contamination', float('nan'))
        flag = (' -- context expectation is contaminated by batter identity '
                f'(corr with mean bat speed = {contam:.2f}), so this score is '
                'partly subtracting the skill it should be measuring'
                if pd.notna(contam) and abs(contam) > 0.4 else '')
        lines.append(f'- **{r["short"]}**: split-half rho {r["reliab_rho_tb_above_expected"]:.3f} '
                     f'vs {r["reliab_rho_mean_tb"]:.3f} for raw mean TB '
                     f'(delta {delta:+.3f}) -- {verdict}.{flag}')
    lines += ['\n## Caveats\n',
              '- MERF assumes Gaussian errors; total bases is a zero-inflated count. '
              'Point estimates (conditional means) are fine, GLL is a convergence '
              'diagnostic only -- judge on held-out R^2 and reliability.',
              '- `tb_above_expected` is measured against the fixed-effect-only '
              'prediction, so it is not shrunk. `random_intercept` is MERF\'s shrunk '
              'version of the same quantity; use it for rankings on low-sample batters.',
              '- field_error scores 0 total bases (standard convention); '
              'sac flies and fielder\'s choices likewise.\n']
    path = os.path.join(OUT_DIR, 'tb_evaluation_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'-> {path}')


if __name__ == '__main__':
    main()
