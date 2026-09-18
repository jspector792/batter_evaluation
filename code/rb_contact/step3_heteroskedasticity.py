"""
step3_heteroskedasticity.py
=============================
RB-Contact% Step 3: does the offset model's residual variance depend
meaningfully on swing difficulty? Decides whether Option 5 (distributional
model) is worth building at all.

Per spec: whiff and contact populations are kept SEPARATE throughout, since
their "true offset" comes from different measurement processes (real
miss_distance for whiffs; launch-angle-derived for contact) -- mixing them
into one residual-variance analysis would conflate measurement-process
differences with genuine swing-difficulty heteroskedasticity.

Decision rule (spec Step 3.4): variance ratio (top predicted-difficulty
quartile vs bottom) < ~2x -> stop, Options 1a/1b suffice. > 2-3x -> build
Option 5 (Step 3b).
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR
from model_utils import CONTINUOUS_RAW

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE, RED, GREEN = '#2563EB', '#DC2626', '#16A34A'


def load_residuals() -> pd.DataFrame:
    df = prepare_data()
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f'{oof_path} not found -- run oof_timing_offset.py first (Step 1c).'
        )
    oof = pd.read_parquet(oof_path)[['row_key', 'predicted_barrel']]
    df = df.merge(oof, on='row_key', how='inner')
    df = df.dropna(subset=['predicted_barrel', 'barrel_distance_v2'])
    df['residual'] = df['barrel_distance_v2'] - df['predicted_barrel']
    df['sq_residual'] = df['residual'] ** 2
    return df


def variance_by_quartile(df: pd.DataFrame, population_label: str) -> dict:
    """Split by predicted_barrel quartile (the 'predicted-difficulty' proxy),
    report variance ratio top vs bottom quartile."""
    df = df.copy()
    df['difficulty_q'] = pd.qcut(df['predicted_barrel'], 4, labels=False, duplicates='drop')
    var_by_q = df.groupby('difficulty_q')['residual'].var()
    n_by_q = df.groupby('difficulty_q')['residual'].size()
    ratio = var_by_q.iloc[-1] / var_by_q.iloc[0]
    print(f'\n[{population_label}] Residual variance by predicted-offset quartile:')
    for q in var_by_q.index:
        print(f'  Q{q+1}: n={n_by_q[q]:,}  var={var_by_q[q]:.4f}  sd={np.sqrt(var_by_q[q]):.4f}')
    print(f'  Variance ratio (top quartile / bottom quartile): {ratio:.2f}x')
    return dict(var_by_quartile=var_by_q.to_dict(), ratio=ratio)


def regress_squared_residuals(df: pd.DataFrame, population_label: str, covariates: list):
    """Simple variance model: regress squared residuals on X_raw covariates.
    A meaningfully nonzero, jointly significant fit indicates heteroskedasticity
    tied to specific swing/pitch characteristics, not just overall model fit."""
    avail = [c for c in covariates if c in df.columns and df[c].notna().all()]
    formula = 'sq_residual ~ ' + ' + '.join(avail)
    model = smf.ols(formula, data=df.dropna(subset=avail + ['sq_residual'])).fit()
    print(f'\n[{population_label}] Variance regression (sq_residual ~ X_raw):')
    print(f'  R^2 = {model.rsquared:.4f}   F-stat p-value = {model.f_pvalue:.2e}')
    return model


def plot_variance_by_quartile(results: dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, res) in zip(axes, results.items()):
        var_by_q = res['var_by_quartile']
        qs = sorted(var_by_q.keys())
        ax.bar([f'Q{q+1}' for q in qs], [var_by_q[q] for q in qs], color=BLUE, alpha=0.85)
        ax.set_title(f'{label}\nratio (top/bottom) = {res["ratio"]:.2f}x', fontweight='bold')
        ax.set_ylabel('Residual variance')
        ax.axhline(var_by_q[qs[0]], color=RED, linestyle='--', linewidth=1,
                   label='bottom-quartile variance')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\n-> {path}')


def main():
    df = load_residuals()
    is_whiff = df['is_contact'] == False

    results = {}
    for label, mask in [('whiff (real miss_distance)', is_whiff),
                        ('contact (launch-angle-derived)', ~is_whiff)]:
        sub = df[mask]
        print(f'\n{"="*70}\n{label}: n={len(sub):,}\n{"="*70}')
        results[label] = variance_by_quartile(sub, label)
        regress_squared_residuals(sub, label, CONTINUOUS_RAW)

    plot_variance_by_quartile(results, os.path.join(OUT_DIR, 'step3_heteroskedasticity.png'))

    # ── Decision rule ────────────────────────────────────────────────────────
    max_ratio = max(r['ratio'] for r in results.values())
    print(f'\n{"="*70}\nDECISION (Step 3.4): max variance ratio across populations = {max_ratio:.2f}x')
    if max_ratio < 2.0:
        decision = 'STOP -- variance roughly flat. Options 1a/1b are sufficient; do NOT build Option 5.'
    elif max_ratio < 3.0:
        decision = 'BORDERLINE -- discuss with domain owner before committing to Option 5.'
    else:
        decision = 'PROCEED to Step 3b (Option 5, distributional model) -- meaningful heteroskedasticity found.'
    print(decision)

    with open(os.path.join(OUT_DIR, 'step3_decision.txt'), 'w') as f:
        f.write(f'Max variance ratio: {max_ratio:.4f}\n{decision}\n')


if __name__ == '__main__':
    main()
