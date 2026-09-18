"""
miss_distance_models.py
=======================
Three mixed-effects model variants predicting miss distance on swinging strikes.

pfx_x is taken as absolute value throughout to control for batter handedness,
so positive values always mean the pitch moved away from the barrel side.

Models
------
1. BASE:
   miss_distance ~ release_speed_c + plate_x + plate_z + abs_pfx_x + pfx_z
                 + (1 | batter) + (1 | batter:pitch_type)

2. WITH_INT_Y:
   miss_distance ~ release_speed_c + plate_x + plate_z + abs_pfx_x + pfx_z
                 + intercept_y
                 + (1 | batter) + (1 | batter:pitch_type)

3. WITH_INT_XY (drops plate_x, adds both intercepts):
   miss_distance ~ release_speed_c + plate_z + abs_pfx_x + pfx_z
                 + intercept_x + intercept_y
                 + (1 | batter) + (1 | batter:pitch_type)

Scoring
-------
miss_dist_score = 1 / mean(|residual|)  for batters with >= 400 AB proxy
Higher score = smaller average unexplained miss = better contact skill.

Outputs (written to OUT_DIR)
----------------------------
  miss_dist_comparison.csv          R² / AIC comparison table
  miss_dist_comparison.png          bar chart
  miss_dist_{label}_summary.png     residual diagnostic plots
  miss_dist_{label}_summary.txt     full lmer summary from R
  miss_dist_scores.csv              per-batter scores (≥400 PA)

Requires: R with lme4, performance installed.
"""

import os, glob, sys, subprocess, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column aliases ─────────────────────────────────────────────────────────────
INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
MISS        = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
IN_PLAY     = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL        = {'foul', 'foul_tip', 'foul_bunt'}
ALL_SWINGS  = IN_PLAY | FOUL | MISS

MIN_PA = 400   # minimum AB proxy for scoring

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

    # ── Standardise release_speed ─────────────────────────────────────────────
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std
    print(f'release_speed centred (mean={rs_mean:.2f}, sd={rs_std:.2f})')

    # ── Absolute pfx_x (handedness-adjusted) ─────────────────────────────────
    # pfx_x is signed: positive = arm-side run. Taking abs() means we always
    # measure magnitude of horizontal break regardless of pitcher hand.
    df['abs_pfx_x'] = df['pfx_x'].abs()

    # ── Absolute pfx_x (handedness-adjusted) ─────────────────────────────────
    df['abs_pfx_x'] = df['pfx_x'].abs()

    # ── plate_x_bat_flip: flip sign for RHB so positive = outer edge ──────────
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    print('plate_x_bat_flip applied (R→negated, L→unchanged)')

    # ── Rename intercepts ─────────────────────────────────────────────────────
    df = df.rename(columns={
        INTERCEPT_X: 'intercept_x',
        INTERCEPT_Y: 'intercept_y',
    })

    # ── String grouping factors ───────────────────────────────────────────────
    for col in ['batter', 'pitch_type']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── Filter to swinging strikes only ──────────────────────────────────────
    sub = df[df['description'].isin(MISS)].copy()
    print(f'Swinging strikes: {len(sub):,}')

    # ── Drop rows missing any modelling variable ──────────────────────────────
    needed = ['miss_distance', 'release_speed_c', 'plate_x_bat_flip', 'plate_z',
              'abs_pfx_x', 'pfx_z', 'intercept_x', 'intercept_y',
              'batter', 'pitch_type']
    sub = sub.dropna(subset=needed).copy()
    print(f'After dropping NA: {len(sub):,}')
    print(f'Unique batters: {sub["batter"].nunique()}, '
          f'pitch types: {sub["pitch_type"].nunique()}')

    # ── PA proxy from full dataset (all pitches, not just misses) ─────────────
    # AB proxy = balls_in_play + strikeouts (same logic as swing_models_via_r)
    df_str = df.copy()
    df_str['batter'] = df_str['batter'].astype(str)
    bip_mask = df_str['description'].isin(IN_PLAY)
    k_mask   = df_str['events'] == 'strikeout'
    bip_ct   = df_str[bip_mask].groupby('batter').size().rename('n_bip')
    k_ct     = df_str[k_mask].groupby('batter').size().rename('n_k')
    pa_proxy = (pd.concat([bip_ct, k_ct], axis=1)
                  .fillna(0)
                  .astype(int)
                  .assign(ab_proxy=lambda x: x['n_bip'] + x['n_k'])
                  .reset_index()[['batter', 'ab_proxy']])
    sub = sub.merge(pa_proxy, on='batter', how='left')
    sub['ab_proxy'] = sub['ab_proxy'].fillna(0)

    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 2. R INTERFACE  (mirrors run_lmer_via_r from swing_models_via_r.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_lmer_via_r(df: pd.DataFrame, outcome: str, formula: str,
                   label: str, out_prefix: str) -> dict | None:
    data_path = os.path.join(OUT_DIR, f'_tmp_{label}_data.csv')
    df.to_csv(data_path, index=False)

    r_script = f'''
library(lme4)
library(performance)

df <- read.csv("{data_path}")

for (col in c("batter", "pitch_type")) {{
  if (col %in% names(df)) df[[col]] <- as.factor(df[[col]])
}}

cat("Fitting: {formula}\\n")
model <- lmer({formula}, data=df, REML=TRUE,
              control=lmerControl(optimizer="bobyqa"))

is_singular <- isSingular(model)
cat("Singular:", is_singular, "\\n")

fitted_vals <- fitted(model)

r2_vals <- tryCatch(
  r2(model),
  error = function(e) list(R2_marginal=NA, R2_conditional=NA)
)

if (is.list(r2_vals)) {{
  marginal_r2   <- r2_vals$R2_marginal
  conditional_r2 <- r2_vals$R2_conditional
}} else {{
  var_fixed      <- var(predict(model, re.form=NA))
  var_total      <- var(df[["{outcome}"]])
  marginal_r2    <- var_fixed / var_total
  conditional_r2 <- 1 - var(residuals(model)) / var_total
}}

aic_val <- AIC(model)

fe <- summary(model)$coefficients
cat("\\nFixed effects:\\n")
print(fe)
cat("\\nVariance components:\\n")
print(VarCorr(model))
cat("\\nMarginal R²:", marginal_r2, "\\n")
cat("Conditional R²:", conditional_r2, "\\n")
cat("AIC:", aic_val, "\\n")

write.csv(data.frame(fitted=fitted_vals), "{out_prefix}_fitted.csv",  row.names=FALSE)
write.csv(fe,                              "{out_prefix}_coefs.csv",   row.names=TRUE)
write.table(
  data.frame(marginal_r2=marginal_r2, conditional_r2=conditional_r2,
             aic=aic_val, singular=is_singular),
  "{out_prefix}_summary.csv", row.names=FALSE, sep=","
)

sink("{out_prefix}_summary.txt")
print(summary(model))
sink()
'''

    r_path = os.path.join(OUT_DIR, f'_tmp_{label}_script.R')
    with open(r_path, 'w') as f:
        f.write(r_script)

    print(f'\n  Running R: {label}')
    try:
        result = subprocess.run(['Rscript', r_path],
                                capture_output=True, text=True,
                                timeout=600, cwd=OUT_DIR)
        print(result.stdout)
        if result.returncode != 0:
            print(f'  R stderr:\n{result.stderr}')
            return None
    except Exception as e:
        print(f'  R execution failed: {e}')
        return None

    try:
        fitted_df   = pd.read_csv(f'{out_prefix}_fitted.csv')
        coefs_df    = pd.read_csv(f'{out_prefix}_coefs.csv', index_col=0)
        summary_df  = pd.read_csv(f'{out_prefix}_summary.csv')

        fitted       = pd.Series(fitted_df['fitted'].values, index=df.index)
        cond_r2      = float(summary_df['conditional_r2'].iloc[0])
        marginal_r2  = float(summary_df['marginal_r2'].iloc[0])
        aic          = float(summary_df['aic'].iloc[0])
        singular     = bool(summary_df['singular'].iloc[0])

        for ext in ['_data.csv', '_script.R', '_fitted.csv',
                    '_coefs.csv', '_summary.csv']:
            try:
                os.remove(os.path.join(OUT_DIR, f'_tmp_{label}{ext}'))
            except OSError:
                pass

        return {
            'label': label, 'formula': formula, 'fitted': fitted,
            'cond_r2': cond_r2, 'marginal_r2': marginal_r2,
            'aic': aic, 'singular': singular, 'coefs': coefs_df,
        }
    except Exception as e:
        print(f'  Failed to read R outputs: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAPLEY R²
# ══════════════════════════════════════════════════════════════════════════════

def compute_shapley_r2(df: pd.DataFrame, outcome: str,
                       predictor_pool: list) -> pd.Series:
    """
    Exact Shapley decomposition of OLS R² across predictor_pool.
    Fits 2^n OLS models; fine for n ≤ 6.
    """
    def ols_r2(preds):
        if not preds:
            return 0.0
        terms = [f'C({p})' if df[p].dtype == object else p for p in preds]
        try:
            return smf.ols(f'{outcome} ~ {" + ".join(terms)}',
                           data=df).fit().rsquared
        except Exception:
            return np.nan

    cache = {
        frozenset(c): ols_r2(list(c))
        for s in range(len(predictor_pool) + 1)
        for c in itertools.combinations(predictor_pool, s)
    }

    shapley = {}
    for p in predictor_pool:
        others  = [q for q in predictor_pool if q != p]
        contribs = [
            cache.get(frozenset(s) | {p}, 0) - cache.get(frozenset(s), 0)
            for sz in range(len(others) + 1)
            for s in itertools.combinations(others, sz)
        ]
        shapley[p] = np.nanmean(contribs)

    return pd.Series(shapley).sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def summary_plots(result: dict, df: pd.DataFrame, fname_prefix: str):
    fitted = result['fitted']
    actual = df['miss_distance']
    resid  = actual - fitted

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Miss Distance – {result["label"]}',
                 fontsize=13, fontweight='bold')

    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.12, s=4, color=BLUE, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    # Predicted vs actual
    ax = axes[0, 1]
    lim = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
    ax.scatter(actual, fitted, alpha=0.12, s=4, color=BLUE, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(actual, fitted)[0, 1]
    ax.set_xlabel('Actual miss distance'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # Fixed-effect coefficients
    ax = axes[1, 0]
    coefs = result['coefs']
    preds = [i for i in coefs.index if i != '(Intercept)']
    if preds:
        y_pos    = np.arange(len(preds))
        ests     = coefs.loc[preds, 'Estimate']
        se       = coefs.loc[preds, 'Std. Error']
        ci_lo    = ests - 1.96 * se
        ci_hi    = ests + 1.96 * se
        ax.barh(y_pos, ests, color=GREEN, alpha=0.7, height=0.5)
        ax.errorbar(ests, y_pos,
                    xerr=[ests - ci_lo, ci_hi - ests],
                    fmt='none', color=DARK, linewidth=1.5, capsize=4)
        ax.axvline(0, color=RED, linewidth=1, linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(preds, fontsize=9)
        ax.set_xlabel('Coefficient')
        ax.set_title('Fixed effects (95% CI)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Residual distribution
    ax = axes[1, 1]
    lo, hi = resid.quantile(0.005), resid.quantile(0.995)
    ax.hist(resid.clip(lo, hi), bins=80,
            color=BLUE, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (σ = {resid.std():.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{fname_prefix}_summary.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def comparison_plot(records: list, fname: str):
    df_rec = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Miss Distance Model Comparison', fontsize=13, fontweight='bold')

    x = np.arange(len(df_rec))

    for ax, col, title in [
        (axes[0], 'cond_r2',  'Conditional R²'),
        (axes[1], 'aic',      'AIC  (lower = better)'),
    ]:
        bars = ax.bar(x, df_rec[col], color=BLUE, edgecolor='white', alpha=0.85)
        for bar, v, sing in zip(bars, df_rec[col], df_rec['singular']):
            label = f'{v:.1f}' if col == 'aic' else f'{v:.4f}'
            if sing:
                label += ' ⚠'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (max(df_rec[col]) - min(df_rec[col])) * 0.01,
                    label, ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df_rec['label'], rotation=20, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. THREE MODEL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_EFFECTS = '(1 | batter) + (1 | batter:pitch_type)'

VARIANTS = [
    (
        'base',
        f'miss_distance ~ release_speed_c + plate_x_bat_flip + plate_z + abs_pfx_x + pfx_z + {RANDOM_EFFECTS}',
        ['release_speed_c', 'plate_x_bat_flip', 'plate_z', 'abs_pfx_x', 'pfx_z'],
    ),
    (
        'with_int_y',
        f'miss_distance ~ release_speed_c + plate_x_bat_flip + plate_z + abs_pfx_x + pfx_z + intercept_y + {RANDOM_EFFECTS}',
        ['release_speed_c', 'plate_x_bat_flip', 'plate_z', 'abs_pfx_x', 'pfx_z', 'intercept_y'],
    ),
    (
        'with_int_xy',
        f'miss_distance ~ release_speed_c + plate_z + abs_pfx_x + pfx_z + intercept_x + intercept_y + {RANDOM_EFFECTS}',
        ['release_speed_c', 'plate_z', 'abs_pfx_x', 'pfx_z', 'intercept_x', 'intercept_y'],
    ),
]


def run_all_variants(sub: pd.DataFrame) -> tuple[dict, list]:
    results = {}
    records = []
    shapley_all = {}

    for label, formula, shap_pool in VARIANTS:
        print(f'\n{"="*70}')
        print(f'VARIANT: {label}')
        print(f'Formula: {formula}')
        print(f'{"="*70}')

        out_pfx = os.path.join(OUT_DIR, f'miss_dist_{label}')
        r = run_lmer_via_r(sub, 'miss_distance', formula, label, out_pfx)

        if r is None:
            print(f'  Skipping {label} — R run failed.')
            continue

        results[label] = r
        records.append({
            'label':      label,
            'formula':    formula,
            'marginal_r2':   r['marginal_r2'],
            'cond_r2':    r['cond_r2'],
            'aic':        r['aic'],
            'singular':   r['singular'],
        })

        # Shapley on the modelling subset
        print(f'\n  Computing Shapley R² for {label}...')
        shap = compute_shapley_r2(sub, 'miss_distance', shap_pool)
        shapley_all[label] = shap
        print(f'  Shapley R²:\n{shap.to_string()}')

        # Save Shapley
        shap_path = os.path.join(OUT_DIR, f'miss_dist_{label}_shapley.csv')
        shap.reset_index().rename(columns={'index': 'predictor', 0: 'shapley_r2'}) \
            .to_csv(shap_path, index=False)
        print(f'  → {shap_path}')

        # Diagnostic plots
        summary_plots(r, sub, f'miss_dist_{label}')

    return results, records, shapley_all


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(sub: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    For each variant, compute per-batter miss_dist_score = 1 / mean(|residual|),
    restricted to batters with ab_proxy >= MIN_PA.

    Returns a wide DataFrame: one row per batter, one score column per variant.
    """
    print(f'\n{"="*70}')
    print(f'SCORING  (minimum {MIN_PA} AB proxy)')
    print('='*70)

    # Identify qualifying batters from the full swinging-strike dataset
    qualifying = (sub.groupby('batter')['ab_proxy']
                     .first()                     # ab_proxy is constant per batter
                     .pipe(lambda s: s[s >= MIN_PA])
                     .index)
    print(f'Qualifying batters: {len(qualifying):,}')

    score_frames = []

    for label, r in results.items():
        resid = (sub['miss_distance'] - r['fitted']).abs()
        frame = (
            pd.DataFrame({'batter': sub['batter'], 'abs_resid': resid})
            .groupby('batter')['abs_resid']
            .mean()
            .loc[lambda s: s.index.isin(qualifying)]
            .rename(f'mean_abs_resid_{label}')
            .reset_index()
        )
        frame[f'miss_dist_score_{label}'] = 1.0 / frame[f'mean_abs_resid_{label}']
        score_frames.append(frame[['batter',
                                    f'mean_abs_resid_{label}',
                                    f'miss_dist_score_{label}']])

    # Merge all variants on batter
    scores = score_frames[0]
    for sf in score_frames[1:]:
        scores = scores.merge(sf, on='batter', how='outer')

    # Summary
    score_cols = [c for c in scores.columns if c.startswith('miss_dist_score_')]
    print(scores[score_cols].describe().round(4).to_string())

    path = os.path.join(OUT_DIR, 'miss_dist_scores.csv')
    scores.to_csv(path, index=False)
    print(f'\n→ Scores written to {path}')

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Miss Distance Model — three variants\n')

    sub = prepare_data()

    results, records, shapley_all = run_all_variants(sub)

    if not results:
        print('\nNo models completed successfully. Exiting.')
        sys.exit(1)

    # Comparison table + plot
    comp_df = pd.DataFrame(records)
    comp_path = os.path.join(OUT_DIR, 'miss_dist_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f'\nModel comparison:\n{comp_df.to_string(index=False)}')
    comparison_plot(records, 'miss_dist_comparison.png')

    # Scores
    scores = compute_scores(sub, results)

    # Pick best model by AIC and flag it
    best = comp_df.loc[comp_df['aic'].idxmin(), 'label']
    print(f'\nBest model by AIC: {best}')
    print(f'All outputs written to {OUT_DIR}')


if __name__ == '__main__':
    main()