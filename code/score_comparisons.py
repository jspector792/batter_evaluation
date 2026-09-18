"""
score_comparison.py
===================
Validation analysis comparing model-derived batter scores against
external benchmarks, plus OLS regression models predicting future xwOBA.

Analyses
--------
1.  Barrel score vs xwOBA (2025)
2.  xwOBA year-over-year stability (2025 → 2026)
3.  Barrel score (2025) vs xwOBA (2026)
4.  Timing score vs Savant on_time_percent
5.  Barrel score vs Savant perfect_percent
6.  OLS prediction of 2026 xwOBA:
      Model A — barrel score alone
      Model B — 2025 xwOBA alone  (baseline)
      Model C — 2025 xwOBA + barrel score
      Model D — 2025 xwOBA + barrel score + timing score
    Each model reports R², adjusted R², AIC, and coefficient table.
    A coefficient plot compares standardised betas across models.

All analyses require MIN_PA plate appearances. Scores are z-scored before
entering regression models so coefficients are comparable across predictors.

Outputs (written to out/diagnostics/)
--------------------------------------
  score_comparison_barrel_vs_xwoba.png
  score_comparison_xwoba_yoy.png
  score_comparison_barrel_yoy.png
  score_comparison_timing_vs_savant.png
  score_comparison_barrel_vs_perfect.png
  score_comparison_prediction_models.png   coefficient plot across OLS models
  score_comparison_prediction_actual.png   predicted vs actual 2026 xwOBA
  score_comparison_summary.csv             correlation results
  score_comparison_ols_summary.csv         OLS model comparison table
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

BARREL_SCORES_PATH = os.path.join(BASE_DIR, 'out', 'final_models',
                                   'contact_v3_scores.csv')
TIMING_SCORES_PATH = os.path.join(BASE_DIR, 'out', 'final_models',
                                   'int_y_merf_scores_v2.csv')
BATTER_STATS_2025  = os.path.join(DATA_DIR, 'batter_stats_2025.csv')
BATTER_STATS_2026  = os.path.join(DATA_DIR, 'batter_stats_2026.csv')
SAVANT_TIMING_PATH = os.path.join(BASE_DIR, 'data',
                                   'bat-tracking-swing-timing.csv')

DIAG_DIR = os.path.join(BASE_DIR, 'out', 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

MIN_PA = 200

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; RED = '#DC2626'; GREEN = '#16A34A'; GRAY = '#6B7280'
PURPLE = '#7C3AED'; ORANGE = '#D97706'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f'  SKIPPED ({label}): file not found at {path}')
        return None
    df = pd.read_csv(path)
    print(f'  Loaded {label}: {len(df):,} rows')
    return df


def correlate(x: pd.Series, y: pd.Series,
              label_x: str, label_y: str) -> dict:
    mask = x.notna() & y.notna()
    n    = mask.sum()
    if n < 10:
        print(f'  WARNING: only {n} overlapping rows for '
              f'{label_x} vs {label_y} — skipping')
        return dict(label_x=label_x, label_y=label_y,
                    r=np.nan, r2=np.nan, p=np.nan, n=n)
    r, p = stats.pearsonr(x[mask], y[mask])
    print(f'  {label_x} vs {label_y}: '
          f'r={r:.3f}, r²={r**2:.3f}, p={p:.4f}, n={n}')
    return dict(label_x=label_x, label_y=label_y,
                r=round(r, 4), r2=round(r**2, 4),
                p=round(p, 6), n=n)


def scatter_with_fit(x: pd.Series, y: pd.Series,
                     xlabel: str, ylabel: str,
                     title: str, out_path: str,
                     color: str = BLUE):
    mask = x.notna() & y.notna()
    xv   = x[mask].values
    yv   = y[mask].values
    slope, intercept, r, p, se = stats.linregress(xv, yv)
    x_line  = np.linspace(xv.min(), xv.max(), 300)
    y_line  = slope * x_line + intercept
    n       = mask.sum()
    x_bar   = xv.mean()
    se_line = se * np.sqrt(1/n + (x_line - x_bar)**2
                           / np.sum((xv - x_bar)**2))
    ci = 1.96 * se_line

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xv, yv, s=14, alpha=0.45, color=color, rasterized=True)
    ax.plot(x_line, y_line, color=RED, linewidth=1.8,
            label=f'slope={slope:.3f}, intercept={intercept:.3f}\n'
                  f'r={r:.3f}, r²={r**2:.3f}, n={n}')
    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color=RED, alpha=0.12, label='95% CI')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND PREP
# ══════════════════════════════════════════════════════════════════════════════

def load_all() -> dict:
    print('\nLoading data...')
    return {
        'barrel':  load_csv(BARREL_SCORES_PATH, 'barrel scores'),
        'timing':  load_csv(TIMING_SCORES_PATH, 'timing scores'),
        'stats25': load_csv(BATTER_STATS_2025,  'batter stats 2025'),
        'stats26': load_csv(BATTER_STATS_2026,  'batter stats 2026'),
        'savant':  load_csv(SAVANT_TIMING_PATH, 'Savant timing leaderboard'),
    }


def prep_barrel(data: dict) -> pd.DataFrame | None:
    df = data['barrel']
    if df is None:
        return None
    df = df.rename(columns={'batter': 'batter_id'})
    df['batter_id'] = df['batter_id'].astype(str)
    return df[['batter_id', 'v3_score', 'n_contact', 'ab_proxy']]


def prep_timing(data: dict) -> pd.DataFrame | None:
    df = data['timing']
    if df is None:
        return None
    df = df.rename(columns={'batter': 'batter_id'})
    df['batter_id'] = df['batter_id'].astype(str)
    return df[['batter_id', 'timing_score', 'n_swings', 'ab_proxy']]


def prep_stats(df: pd.DataFrame | None,
               year_label: str) -> pd.DataFrame | None:
    if df is None:
        return None
    xwoba_col = next((c for c in ['xst_woba', 'xst_xwoba', 'xwoba', 'xwOBA']
                      if c in df.columns), None)
    if xwoba_col is None:
        print(f'  WARNING: xwOBA not found in {year_label}. '
              f'wOBA-like cols: {[c for c in df.columns if "woba" in c.lower()]}')
        return None
    pa_col = next((c for c in ['fg_PA', 'PA', 'br_PA'] if c in df.columns), None)
    out = df[['batter_id', xwoba_col]].copy()
    out = out.rename(columns={xwoba_col: f'xwoba_{year_label}'})
    out['batter_id'] = out['batter_id'].astype(str).str.strip()
    if pa_col:
        out = out[df[pa_col] >= MIN_PA].copy()
        print(f'  {year_label} after PA >= {MIN_PA}: {len(out):,} batters')
    return out.dropna(subset=[f'xwoba_{year_label}'])


def prep_savant(data: dict) -> pd.DataFrame | None:
    df = data['savant']
    if df is None:
        return None
    id_col = next((c for c in ['id', 'player_id', 'batter_id']
                   if c in df.columns), None)
    if id_col is None:
        print(f'  WARNING: no ID column found. Columns: {df.columns.tolist()}')
        return None
    keep = [id_col]
    for col in ['on_time_percent', 'perfect_percent']:
        if col in df.columns:
            keep.append(col)
        else:
            print(f'  WARNING: {col!r} not in Savant data')
    out = df[keep].rename(columns={id_col: 'batter_id'}).copy()
    out['batter_id'] = out['batter_id'].astype(str).str.strip()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSES (1–5)
# ══════════════════════════════════════════════════════════════════════════════

def analysis_barrel_vs_xwoba(barrel, stats25, records):
    print('\n── Analysis 1: Barrel score vs xwOBA (2025) ──')
    if barrel is None or stats25 is None:
        print('  Skipped'); return
    m = barrel.merge(stats25, on='batter_id', how='inner')
    print(f'  Merged: {len(m):,} batters')
    records.append(correlate(m['v3_score'], m['xwoba_2025'],
                             'barrel_score', 'xwoba_2025'))
    scatter_with_fit(m['v3_score'], m['xwoba_2025'],
                     'Barrel placement score (V3)', 'xwOBA 2025',
                     'Barrel placement score vs xwOBA (2025)',
                     os.path.join(DIAG_DIR,
                                  'score_comparison_barrel_vs_xwoba.png'))


def analysis_xwoba_yoy(stats25, stats26, records):
    print('\n── Analysis 2: xwOBA year-over-year (2025 → 2026) ──')
    if stats25 is None or stats26 is None:
        print('  Skipped: 2026 stats not available'); return
    m = stats25.merge(stats26, on='batter_id', how='inner')
    if 'xwoba_2026' not in m.columns:
        print('  Skipped: xwoba_2026 missing'); return
    print(f'  Merged: {len(m):,} batters')
    records.append(correlate(m['xwoba_2025'], m['xwoba_2026'],
                             'xwoba_2025', 'xwoba_2026'))
    scatter_with_fit(m['xwoba_2025'], m['xwoba_2026'],
                     'xwOBA 2025', 'xwOBA 2026',
                     'xwOBA year-over-year stability (2025 → 2026)',
                     os.path.join(DIAG_DIR,
                                  'score_comparison_xwoba_yoy.png'),
                     color=GREEN)


def analysis_barrel_yoy(barrel, stats26, records):
    print('\n── Analysis 3: Barrel score (2025) vs xwOBA (2026) ──')
    if barrel is None or stats26 is None:
        print('  Skipped: 2026 stats not available'); return
    m = barrel.merge(stats26, on='batter_id', how='inner')
    if 'xwoba_2026' not in m.columns:
        print('  Skipped: xwoba_2026 missing'); return
    print(f'  Merged: {len(m):,} batters')
    records.append(correlate(m['v3_score'], m['xwoba_2026'],
                             'barrel_score_2025', 'xwoba_2026'))
    scatter_with_fit(m['v3_score'], m['xwoba_2026'],
                     'Barrel placement score (V3, 2025)', 'xwOBA 2026',
                     'Barrel score (2025) vs xwOBA (2026)',
                     os.path.join(DIAG_DIR,
                                  'score_comparison_barrel_yoy.png'))


def analysis_timing_vs_savant(timing, savant, records):
    print('\n── Analysis 4: Timing score vs Savant on_time_percent ──')
    if timing is None or savant is None or 'on_time_percent' not in savant.columns:
        print('  Skipped'); return
    m = timing.merge(savant[['batter_id', 'on_time_percent']],
                     on='batter_id', how='inner')
    print(f'  Merged: {len(m):,} batters')
    records.append(correlate(m['timing_score'], m['on_time_percent'],
                             'timing_score', 'on_time_percent'))
    scatter_with_fit(m['timing_score'], m['on_time_percent'],
                     'Timing score (int_y MERF)',
                     'On-time % (Baseball Savant)',
                     'Timing score vs Savant on-time %',
                     os.path.join(DIAG_DIR,
                                  'score_comparison_timing_vs_savant.png'),
                     color=GREEN)


def analysis_barrel_vs_perfect(barrel, savant, records):
    print('\n── Analysis 5: Barrel score vs Savant perfect_percent ──')
    if barrel is None or savant is None or \
            'perfect_percent' not in savant.columns:
        print('  Skipped'); return
    m = barrel.merge(savant[['batter_id', 'perfect_percent']],
                     on='batter_id', how='inner')
    print(f'  Merged: {len(m):,} batters')
    records.append(correlate(m['v3_score'], m['perfect_percent'],
                             'barrel_score', 'perfect_percent'))
    scatter_with_fit(m['v3_score'], m['perfect_percent'],
                     'Barrel placement score (V3)',
                     'Perfect contact % (Baseball Savant)',
                     'Barrel score vs Savant perfect contact %',
                     os.path.join(DIAG_DIR,
                                  'score_comparison_barrel_vs_perfect.png'))


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: OLS PREDICTION MODELS
# ══════════════════════════════════════════════════════════════════════════════

def analysis_predict_xwoba(barrel: pd.DataFrame | None,
                            timing: pd.DataFrame | None,
                            stats25: pd.DataFrame | None,
                            stats26: pd.DataFrame | None,
                            ols_records: list):
    """
    Four OLS models predicting 2026 xwOBA:
      A: barrel_score_z alone
      B: xwoba_2025_z alone  (year-over-year baseline)
      C: xwoba_2025_z + barrel_score_z
      D: xwoba_2025_z + barrel_score_z + timing_score_z

    All predictors are z-scored so coefficients are directly comparable.
    Reports R², adj-R², AIC, and a coefficient table per model.
    Also produces:
      - coefficient plot (standardised betas + 95% CI across models)
      - predicted vs actual scatter for the best model by adj-R²
    """
    print('\n── Analysis 6: OLS prediction of 2026 xwOBA ──')

    if stats26 is None:
        print('  Skipped: 2026 stats not available')
        return

    if barrel is None or stats25 is None:
        print('  Skipped: barrel scores or 2025 stats missing')
        return

    # ── Build modelling dataset ────────────────────────────────────────────────
    base = stats25.merge(stats26, on='batter_id', how='inner')
    base = base.merge(barrel[['batter_id', 'v3_score']], on='batter_id',
                      how='inner')
    if timing is not None:
        base = base.merge(timing[['batter_id', 'timing_score']],
                          on='batter_id', how='left')
    else:
        base['timing_score'] = np.nan

    base = base.dropna(subset=['xwoba_2025', 'xwoba_2026', 'v3_score']).copy()
    print(f'  Modelling dataset: {len(base):,} batters')

    if len(base) < 20:
        print('  Skipped: fewer than 20 batters in common')
        return

    # ── Z-score all predictors ─────────────────────────────────────────────────
    # Standardising puts all coefficients on the same scale so we can compare
    # "one SD change in barrel score" vs "one SD change in xwOBA" directly.
    for col in ['xwoba_2025', 'v3_score', 'timing_score']:
        if col in base.columns and base[col].notna().sum() > 1:
            base[f'{col}_z'] = (
                (base[col] - base[col].mean()) / base[col].std()
            )
        else:
            base[f'{col}_z'] = np.nan

    # ── Fit models ────────────────────────────────────────────────────────────
    model_specs = [
        ('A: barrel alone',
         'xwoba_2026 ~ v3_score_z',
         ['v3_score_z']),
        ('B: xwOBA 2025 alone (baseline)',
         'xwoba_2026 ~ xwoba_2025_z',
         ['xwoba_2025_z']),
        ('C: xwOBA 2025 + barrel',
         'xwoba_2026 ~ xwoba_2025_z + v3_score_z',
         ['xwoba_2025_z', 'v3_score_z']),
        ('D: xwOBA 2025 + barrel + timing',
         'xwoba_2026 ~ xwoba_2025_z + v3_score_z + timing_score_z',
         ['xwoba_2025_z', 'v3_score_z', 'timing_score_z']),
    ]

    fitted_models = {}
    coef_rows     = []   # for coefficient plot

    for label, formula, predictors in model_specs:
        # Drop rows missing any predictor for this model
        needed = ['xwoba_2026'] + predictors
        sub    = base.dropna(subset=needed)

        if len(sub) < 20:
            print(f'  [{label}] Skipped: only {len(sub)} complete rows')
            continue

        ols  = smf.ols(formula, data=sub).fit()
        n    = int(ols.nobs)
        r2   = ols.rsquared
        r2a  = ols.rsquared_adj
        aic  = ols.aic

        print(f'\n  [{label}]')
        print(f'    n={n}  R²={r2:.4f}  adj-R²={r2a:.4f}  AIC={aic:.1f}')
        print(f'    Coefficients:')
        for pred in predictors:
            coef = ols.params.get(pred, np.nan)
            se   = ols.bse.get(pred, np.nan)
            pval = ols.pvalues.get(pred, np.nan)
            sig  = '***' if pval < 0.001 else ('**' if pval < 0.01
                   else ('*' if pval < 0.05 else ''))
            print(f'      {pred}: β={coef:.4f}  SE={se:.4f}  '
                  f'p={pval:.4f} {sig}')
            coef_rows.append(dict(
                model=label, predictor=pred,
                beta=coef, se=se, p=pval,
                ci_lo=coef - 1.96*se, ci_hi=coef + 1.96*se,
            ))

        ols_records.append(dict(
            model=label, formula=formula, n=n,
            r2=round(r2, 4), adj_r2=round(r2a, 4), aic=round(aic, 2),
        ))
        fitted_models[label] = (ols, sub)

    if not fitted_models:
        print('  No models completed')
        return

    # ── Save OLS summary ──────────────────────────────────────────────────────
    ols_df = pd.DataFrame(ols_records)
    ols_path = os.path.join(DIAG_DIR, 'score_comparison_ols_summary.csv')
    ols_df.to_csv(ols_path, index=False)
    print(f'\n  OLS summary → {ols_path}')

    # ── Coefficient plot ──────────────────────────────────────────────────────
    # Shows standardised beta ± 95% CI for each predictor across all models.
    # Makes it easy to see whether barrel score adds beyond xwOBA and in
    # which direction, and whether the effect is consistent across models.
    coef_df = pd.DataFrame(coef_rows)

    pred_labels = {
        'v3_score_z':     'Barrel score',
        'xwoba_2025_z':   'xwOBA 2025',
        'timing_score_z': 'Timing score',
    }
    coef_df['predictor_label'] = coef_df['predictor'].map(pred_labels)
    coef_df['model_short'] = coef_df['model'].str.split(':').str[0]

    model_list = coef_df['model_short'].unique().tolist()
    pred_list  = [v for k, v in pred_labels.items()
                  if k in coef_df['predictor'].values]
    colors     = {m: c for m, c in zip(model_list,
                                        [BLUE, GREEN, PURPLE, ORANGE])}

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(model_list)
    offsets  = np.linspace(-0.25, 0.25, n_models)

    for i, (model_short, offset) in enumerate(zip(model_list, offsets)):
        sub_c = coef_df[coef_df['model_short'] == model_short]
        for _, row in sub_c.iterrows():
            x_pos = pred_list.index(row['predictor_label']) + offset
            ax.errorbar(x_pos, row['beta'],
                        yerr=[[row['beta'] - row['ci_lo']],
                               [row['ci_hi'] - row['beta']]],
                        fmt='o', color=colors[model_short],
                        markersize=7, linewidth=2, capsize=4,
                        label=model_short if row.name == sub_c.index[0]
                              else '_nolegend_')

    ax.axhline(0, color=RED, linewidth=1.0, linestyle='--')
    ax.set_xticks(range(len(pred_list)))
    ax.set_xticklabels(pred_list, fontsize=11)
    ax.set_ylabel('Standardised beta (± 95% CI)', fontsize=11)
    ax.set_title('OLS coefficients predicting 2026 xwOBA\n'
                 '(standardised predictors — coefficients directly comparable)',
                 fontweight='bold', fontsize=12)
    ax.legend(title='Model', fontsize=9, title_fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    coef_path = os.path.join(DIAG_DIR,
                              'score_comparison_prediction_models.png')
    fig.savefig(coef_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {coef_path}')

    # ── Predicted vs actual for best model by adj-R² ──────────────────────────
    if ols_records:
        best_label = max(ols_records, key=lambda x: x['adj_r2'])['model']
        if best_label in fitted_models:
            best_ols, best_sub = fitted_models[best_label]
            y_pred = best_ols.fittedvalues
            y_true = best_sub['xwoba_2026']

            r, _ = stats.pearsonr(y_pred, y_true)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_true, y_pred, s=18, alpha=0.5,
                       color=BLUE, rasterized=True)
            lim = [min(y_true.min(), y_pred.min()),
                   max(y_true.max(), y_pred.max())]
            ax.plot(lim, lim, color=RED, linewidth=1.5,
                    linestyle='--', label='Perfect prediction')
            ax.set_xlabel('Actual xwOBA 2026', fontsize=11)
            ax.set_ylabel('Predicted xwOBA 2026', fontsize=11)
            ax.set_title(f'Predicted vs actual xwOBA 2026\n'
                         f'Best model: {best_label}  '
                         f'(r={r:.3f}, adj-R²='
                         f'{best_ols.rsquared_adj:.3f})',
                         fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            pred_path = os.path.join(DIAG_DIR,
                                      'score_comparison_prediction_actual.png')
            fig.savefig(pred_path, dpi=130, bbox_inches='tight')
            plt.close()
            print(f'  → {pred_path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Score Comparison Analysis')
    print('='*60)

    data        = load_all()
    corr_records = []
    ols_records  = []

    barrel  = prep_barrel(data)
    timing  = prep_timing(data)
    stats25 = prep_stats(data['stats25'], '2025')
    stats26 = prep_stats(data['stats26'], '2026')
    savant  = prep_savant(data)

    analysis_barrel_vs_xwoba(barrel,  stats25, corr_records)
    analysis_xwoba_yoy(stats25, stats26, corr_records)
    analysis_barrel_yoy(barrel,  stats26, corr_records)
    analysis_timing_vs_savant(timing,  savant,  corr_records)
    analysis_barrel_vs_perfect(barrel, savant,  corr_records)
    analysis_predict_xwoba(barrel, timing, stats25, stats26, ols_records)

    # ── Correlation summary ───────────────────────────────────────────────────
    if corr_records:
        corr_df = pd.DataFrame(corr_records)
        corr_path = os.path.join(DIAG_DIR, 'score_comparison_summary.csv')
        corr_df.to_csv(corr_path, index=False)
        print(f'\nCorrelation summary:\n{corr_df.to_string(index=False)}')
        print(f'→ {corr_path}')

    # ── OLS summary ───────────────────────────────────────────────────────────
    if ols_records:
        ols_df = pd.DataFrame(ols_records)
        print(f'\nOLS model comparison:\n{ols_df.to_string(index=False)}')

    print(f'\nAll outputs in {DIAG_DIR}')


if __name__ == '__main__':
    main()