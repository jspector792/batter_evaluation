"""
finalize_reports.py
=====================
Consolidates evaluation_report.md to include Step 3 (heteroskedasticity)
and Step 4 (batter random-effect A/B test) results inline, per spec's
deliverables list ("evaluation_report.md -- log loss/Brier/calibration
tables for all model variants (Step 2), heteroskedasticity check results
(Step 3), batter random-effect A/B results with swing-count-bucketed
breakdown (Step 4)"). Also renders the Step 2 calibration curve plots --
the underlying data was computed by model_utils.evaluate() and pickled by
step1_2_contact_models.py, but never actually plotted.

Also merges oof_contact_probs.parquet into oof_predictions.parquet so the
"oof_predictions.parquet" deliverable is the single file the spec describes
(predicted_timing, predicted_offset, AND final contact probabilities),
rather than two separate files.
"""

import os, sys, pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import OUT_DIR

sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, PURPLE = '#2563EB', '#DC2626', '#16A34A', '#7C3AED'


def plot_calibration_curves(results: dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    colors = {'raw_logistic': BLUE, 'raw_gbm': RED,
             'hybrid_logistic': GREEN, 'hybrid_gbm': PURPLE}
    for ax, label in zip(axes.flat, ['raw_logistic', 'raw_gbm', 'hybrid_logistic', 'hybrid_gbm']):
        if label not in results:
            ax.set_visible(False)
            continue
        fold0 = results[label][0]['precal']
        ax.plot(fold0['calib_mean_pred'], fold0['calib_frac_pos'], 'o-',
                color=colors[label], label=f'{label} (fold 0)')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Mean predicted P(contact)')
        ax.set_ylabel('Observed contact rate')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'-> {path}')


def main():
    pkl_path = os.path.join(OUT_DIR, '_step1_2_raw_results.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        plot_calibration_curves(results, os.path.join(OUT_DIR, 'step2_calibration_curves.png'))
    else:
        print(f'WARNING: {pkl_path} not found -- run step1_2_contact_models.py first. Skipping calibration plots.')

    # ── Consolidate evaluation_report.md ────────────────────────────────────
    eval_path = os.path.join(OUT_DIR, 'evaluation_report.md')
    with open(eval_path) as f:
        existing = f.read()

    parts = ['# RB-Contact% -- Consolidated Evaluation Report\n']
    parts.append(existing)

    parts.append('\n## Step 2 -- Calibration curves\n\n')
    parts.append('See step2_calibration_curves.png (fold-0 calibration for each variant).\n')

    decision_path = os.path.join(OUT_DIR, 'step3_decision.txt')
    if os.path.exists(decision_path):
        with open(decision_path) as f:
            decision = f.read()
        parts.append('\n## Step 3 -- Heteroskedasticity check\n\n')
        parts.append('See step3_heteroskedasticity.png for the variance-by-quartile plot.\n\n')
        parts.append(f'```\n{decision}```\n')

    step4_path = os.path.join(OUT_DIR, 'step4_batter_re_ab_test.md')
    if os.path.exists(step4_path):
        with open(step4_path) as f:
            step4 = f.read()
        parts.append('\n## Step 4 -- Batter random-effect A/B test\n\n')
        parts.append(step4)
        parts.append('\n')

    with open(eval_path, 'w') as f:
        f.write('\n'.join(parts))
    print(f'-> consolidated {eval_path}')

    # ── Merge oof_contact_probs.parquet into oof_predictions.parquet ────────
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    probs_path = os.path.join(OUT_DIR, 'oof_contact_probs.parquet')
    if os.path.exists(oof_path) and os.path.exists(probs_path):
        oof = pd.read_parquet(oof_path)
        probs = pd.read_parquet(probs_path)[['row_key', 'p_contact']]
        merged = oof.merge(probs, on='row_key', how='left')
        merged = merged.rename(columns={'predicted_int_y': 'predicted_timing',
                                        'predicted_barrel': 'predicted_offset'})
        merged.to_parquet(oof_path, index=False)
        print(f'-> merged final contact probabilities into {oof_path} '
              f'({merged["p_contact"].notna().sum():,} / {len(merged):,} rows have p_contact)')
    else:
        print('WARNING: could not merge -- one of oof_predictions.parquet / oof_contact_probs.parquet missing')


if __name__ == '__main__':
    main()
