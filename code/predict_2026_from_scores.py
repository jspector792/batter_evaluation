"""
predict_2026_from_scores.py
============================
Central question: do our derived hitter-skill scores (timing, contact/barrel
placement, miss-distance) add predictive value for a batter's 2026 stats
beyond just using their 2025 stats?

For every major statistical category, fits two models predicting the 2026
value from 2025 data:

  Model B (baseline):  stat_2026 ~ stat_2025
  Model C (scores):    stat_2026 ~ stat_2025 + timing_score + contact_score
                                              + miss_dist_score

All predictors are z-scored. Reports in-sample R²/adj-R² for both models
plus 5-fold cross-validated R² (the honest number given n ~ a few hundred
batters and 3 extra predictors) so an in-sample R² bump isn't mistaken for
real generalization.

Score inputs
------------
  timing_score       int_y_merf_scores_v2.csv           (merf_int_y.py)
  contact_score       contact_v3_scores.csv  (v3_score)  (merf_variants_v3.py)
                       — the contact metric currently wired into score_comparisons.py
  barrel_score        batter_scores_combined.csv (barrel_placement_score)
                       — final_merf_models.py's frozen barrel-distance MERF,
                       tested as an alternative contact metric
  miss_dist_score     miss_dist_scores.csv (miss_dist_score_with_int_y)
                       — best-AIC variant from miss_distance_models.py

Outputs (written to out/predictive_value/)
-------------------------------------------
  predict_2026_summary.csv     one row per stat category, all metrics
  predict_2026_r2_comparison.png   baseline vs scores-augmented R² (CV)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FM_DIR   = os.path.join(BASE_DIR, 'out', 'final_models')
OUT_DIR  = os.path.join(BASE_DIR, 'out', 'predictive_value')
os.makedirs(OUT_DIR, exist_ok=True)

MIN_PA   = 200
N_FOLDS  = 5
SEED     = 42

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; RED = '#DC2626'; GREEN = '#16A34A'; GRAY = '#6B7280'

# ── Target statistical categories ───────────────────────────────────────────────
# (label, column in batter_stats, higher-is-better ignored -- just prediction)
DIRECT_TARGETS = [
    ('BA',              'br_BA'),
    ('OBP',             'br_OBP'),
    ('SLG',             'br_SLG'),
    ('OPS',             'br_OPS'),
    ('wOBA',            'xst_woba'),
    ('xwOBA',           'xst_est_woba'),
    ('xBA',             'xst_est_ba'),
    ('xSLG',            'xst_est_slg'),
    ('avg_exit_velo',   'ev_avg_hit_speed'),
    ('max_exit_velo',   'ev_max_hit_speed'),
    ('barrel_pct',      'ev_brl_percent'),
    ('hard_hit_pct',    'pct_hard_hit_percent'),
    ('k_pct',           'pct_k_percent'),
    ('bb_pct',          'pct_bb_percent'),
    ('whiff_pct',       'pct_whiff_percent'),
    ('chase_pct',       'pct_chase_percent'),
]

SCORE_COLS = ['timing_score', 'contact_score', 'barrel_score', 'miss_dist_score']


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_stats(year: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, f'batter_stats_{year}.csv'))
    df['batter_id'] = df['batter_id'].astype(str).str.strip()

    # ISO isn't a raw column -- derive it
    df['ISO'] = df['br_SLG'] - df['br_BA']

    df = df.rename(columns={col: f'{label}_{year}'
                             for label, col in DIRECT_TARGETS})
    df[f'ISO_{year}'] = df['ISO']

    pa_col = next((c for c in ['br_PA'] if c in df.columns), None)
    keep = ['batter_id'] + [f'{label}_{year}' for label, _ in DIRECT_TARGETS] + [f'ISO_{year}']
    if pa_col:
        keep.append(pa_col)
        df = df.rename(columns={pa_col: f'PA_{year}'})
        keep[-1] = f'PA_{year}'
    return df[keep]


def load_scores() -> pd.DataFrame:
    timing = pd.read_csv(os.path.join(FM_DIR, 'int_y_merf_scores_v2.csv'))
    timing = timing.rename(columns={'batter': 'batter_id'})[['batter_id', 'timing_score']]

    contact = pd.read_csv(os.path.join(FM_DIR, 'contact_v3_scores.csv'))
    contact = contact.rename(columns={'batter': 'batter_id', 'v3_score': 'contact_score'})
    contact = contact[['batter_id', 'contact_score']]

    barrel = pd.read_csv(os.path.join(FM_DIR, 'batter_scores_combined.csv'))
    barrel = barrel.rename(columns={'batter': 'batter_id'})[['batter_id', 'barrel_placement_score']]
    barrel = barrel.rename(columns={'barrel_placement_score': 'barrel_score'})

    miss = pd.read_csv(os.path.join(FM_DIR, 'miss_dist_scores.csv'))
    miss = miss.rename(columns={'batter': 'batter_id'})
    miss = miss[['batter_id', 'miss_dist_score_with_int_y']]
    miss = miss.rename(columns={'miss_dist_score_with_int_y': 'miss_dist_score'})

    for df in [timing, contact, barrel, miss]:
        df['batter_id'] = df['batter_id'].astype(str).str.strip()

    scores = timing.merge(contact, on='batter_id', how='outer') \
                   .merge(barrel, on='batter_id', how='outer') \
                   .merge(miss, on='batter_id', how='outer')
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL COMPARISON FOR ONE STAT CATEGORY
# ══════════════════════════════════════════════════════════════════════════════

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def cv_r2(X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int) -> float:
    """Out-of-fold R^2 for an OLS model with predictors X."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in kf.split(X):
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_all.append(model.predict(X.iloc[test_idx]))
        y_true_all.append(y.iloc[test_idx].values)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return r2_score(y_true_all, y_pred_all)


def compare_one_stat(df: pd.DataFrame, label: str) -> dict | None:
    col25, col26 = f'{label}_2025', f'{label}_2026'
    needed = [col25, col26] + SCORE_COLS
    sub = df.dropna(subset=needed).copy()
    n = len(sub)
    if n < 30:
        print(f'  [{label}] Skipped: only {n} complete rows')
        return None

    sub[f'{col25}_z'] = zscore(sub[col25])
    for c in SCORE_COLS:
        sub[f'{c}_z'] = zscore(sub[c])

    # ── In-sample OLS ──────────────────────────────────────────────────────────
    baseline_formula = f'{col26} ~ {col25}_z'
    full_formula = (f'{col26} ~ {col25}_z + ' +
                     ' + '.join(f'{c}_z' for c in SCORE_COLS))

    m_base = smf.ols(baseline_formula, data=sub).fit()
    m_full = smf.ols(full_formula, data=sub).fit()

    # Likelihood-ratio-ish nested F-test (baseline vs full)
    f_test = m_full.compare_f_test(m_base)  # (fvalue, pvalue, df_diff)

    # ── Cross-validated R² ───────────────────────────────────────────────────────
    X_base = sub[[f'{col25}_z']]
    X_full = sub[[f'{col25}_z'] + [f'{c}_z' for c in SCORE_COLS]]
    y = sub[col26]

    cv_base = cv_r2(X_base, y, N_FOLDS, SEED)
    cv_full = cv_r2(X_full, y, N_FOLDS, SEED)

    return dict(
        stat=label, n=n,
        r2_baseline=round(m_base.rsquared, 4),
        r2_full=round(m_full.rsquared, 4),
        adj_r2_baseline=round(m_base.rsquared_adj, 4),
        adj_r2_full=round(m_full.rsquared_adj, 4),
        cv_r2_baseline=round(cv_base, 4),
        cv_r2_full=round(cv_full, 4),
        cv_delta=round(cv_full - cv_base, 4),
        f_pvalue=round(f_test[1], 4),
        timing_beta=round(m_full.params.get('timing_score_z', np.nan), 4),
        timing_p=round(m_full.pvalues.get('timing_score_z', np.nan), 4),
        contact_beta=round(m_full.params.get('contact_score_z', np.nan), 4),
        contact_p=round(m_full.pvalues.get('contact_score_z', np.nan), 4),
        barrel_beta=round(m_full.params.get('barrel_score_z', np.nan), 4),
        barrel_p=round(m_full.pvalues.get('barrel_score_z', np.nan), 4),
        miss_dist_beta=round(m_full.params.get('miss_dist_score_z', np.nan), 4),
        miss_dist_p=round(m_full.pvalues.get('miss_dist_score_z', np.nan), 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PA-CUTOFF SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

PA_SWEEP = [100, 150, 200, 250, 300]


def pa_sensitivity(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Re-run every stat's comparison at several MIN_PA cutoffs. With n in the
    hundreds and 4 extra predictors, a result that only appears at one cutoff
    and flips sign at others is more likely sample-size noise than signal --
    this table is how we tell the difference.
    """
    records = []
    for min_pa in PA_SWEEP:
        sub = df.copy()
        if 'PA_2025' in sub.columns:
            sub = sub[sub['PA_2025'] >= min_pa]
        if 'PA_2026' in sub.columns:
            sub = sub[sub['PA_2026'] >= min_pa]
        for label in labels:
            result = compare_one_stat(sub, label)
            if result:
                records.append(dict(min_pa=min_pa, stat=label,
                                     n=result['n'], cv_delta=result['cv_delta']))
    return pd.DataFrame(records)


def plot_pa_sensitivity(sens: pd.DataFrame, headline_stats: list, path: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = sns.color_palette('tab10', n_colors=len(headline_stats))
    for stat, color in zip(headline_stats, colors):
        sub = sens[sens['stat'] == stat].sort_values('min_pa')
        if sub.empty:
            continue
        ax.plot(sub['min_pa'], sub['cv_delta'], marker='o', color=color, label=stat)
    ax.axhline(0, color=GRAY, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Minimum PA cutoff (both seasons)')
    ax.set_ylabel('CV R² delta (full model − baseline)')
    ax.set_title('Stability of the scores’ predictive lift across PA cutoffs',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(summary: pd.DataFrame, path: str):
    df = summary.sort_values('cv_delta', ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(df))))
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df['cv_r2_baseline'], color=GRAY, alpha=0.6, height=0.35,
            label='Baseline (2025 stat only)')
    ax.barh(y_pos + 0.35, df['cv_r2_full'], color=BLUE, alpha=0.85, height=0.35,
            label='+ timing/contact/barrel/miss-dist scores')

    ax.set_yticks(y_pos + 0.175)
    ax.set_yticklabels(df['stat'])
    ax.set_xlabel('5-fold CV R² predicting 2026 value')
    ax.set_title('Predicting 2026 stats: raw 2025 stat alone vs + derived scores',
                 fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color=RED, linewidth=1, linestyle='--')
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Predicting 2026 stats from 2025 stats + derived scores')
    print('=' * 70)

    stats25 = load_stats('2025')
    stats26 = load_stats('2026')
    scores  = load_scores()

    print(f'2025 batters: {len(stats25):,}   2026 batters: {len(stats26):,}   '
          f'scored batters: {len(scores):,}')

    df = (stats25.merge(stats26, on='batter_id', how='inner')
                 .merge(scores, on='batter_id', how='inner'))

    if 'PA_2025' in df.columns:
        df = df[df['PA_2025'] >= MIN_PA]
    if 'PA_2026' in df.columns:
        df = df[df['PA_2026'] >= MIN_PA]
    print(f'After PA >= {MIN_PA} in both seasons and score merge: {len(df):,} batters\n')

    labels = [label for label, _ in DIRECT_TARGETS] + ['ISO']

    records = []
    for label in labels:
        result = compare_one_stat(df, label)
        if result:
            records.append(result)
            print(f'  [{label:14s}] n={result["n"]:4d}  '
                  f'CV R² base={result["cv_r2_baseline"]:.4f} → '
                  f'full={result["cv_r2_full"]:.4f}  '
                  f'(Δ={result["cv_delta"]:+.4f}, F-test p={result["f_pvalue"]:.4f})')

    summary = pd.DataFrame(records)
    path = os.path.join(OUT_DIR, 'predict_2026_summary.csv')
    summary.to_csv(path, index=False)
    print(f'\nSummary → {path}')

    plot_summary(summary, os.path.join(OUT_DIR, 'predict_2026_r2_comparison.png'))

    # ── PA-cutoff sensitivity ────────────────────────────────────────────────────
    print(f'\nRunning PA-cutoff sensitivity sweep {PA_SWEEP} …')
    full_merged = (stats25.merge(stats26, on='batter_id', how='inner')
                          .merge(scores, on='batter_id', how='inner'))
    sens = pa_sensitivity(full_merged, labels)
    sens_path = os.path.join(OUT_DIR, 'predict_2026_pa_sensitivity.csv')
    sens.to_csv(sens_path, index=False)
    print(f'Sensitivity table → {sens_path}')

    headline = ['SLG', 'OPS', 'ISO', 'xSLG', 'avg_exit_velo', 'max_exit_velo']
    plot_pa_sensitivity(sens, headline,
                        os.path.join(OUT_DIR, 'predict_2026_pa_sensitivity.png'))

    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
