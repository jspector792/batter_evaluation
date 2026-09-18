"""
step3b_option5_distributional.py
===================================
RB-Contact% Step 3b (Option 5): triggered by step3_heteroskedasticity.py
finding 21x/2.7x variance ratios (whiff/contact) -- well above the spec's
2-3x threshold for skipping this step.

Fits quantile regression (HistGradientBoostingRegressor(loss='quantile'))
for offset (barrel_distance_v2) at a grid of quantile levels, out-of-fold,
using the SAME pre-outcome X_raw-style features as the existing offset
model (BARREL_CONFIG) -- never the true offset or the contact label. Then
integrates each swing's fitted quantile curve against the KNOWN physical
threshold C (bat/ball radius sum, 2.755in, from final_merf_models.py) to get
p_i = P(offset_i < C | X_raw).

Why this isn't the tautology the domain owner warned about: true
barrel_distance_v2 < C holds for EVERY contact row and >= C for EVERY whiff
row BY CONSTRUCTION (miss_distance + C >= C for whiffs; |C*sin(...)| <= C
for contact) -- so P(contact | TRUE offset) would indeed be a deterministic
step function, exactly the forbidden move. But this model only ever
conditions on X_raw (pre-outcome kinematics) to predict a DISTRIBUTION over
offset, then applies the fixed constant C -- it never sees the true offset
or the contact label. That's the same non-leakage argument the spec already
accepts for Option 4's predicted_offset feature, just used more directly
(integrating the known deterministic offset->contact relationship instead
of learning it via a second black-box classifier).
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR, C as BARREL_THRESHOLD

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE, RED, GREEN = '#2563EB', '#DC2626', '#16A34A'

# Same feature set as BARREL_CONFIG in final_merf_models.py/oof_timing_offset.py --
# intentionally NOT the contact-classifier's X_raw; this model predicts the
# offset distribution itself, matching the existing offset model's own inputs.
OFFSET_FEATURES = ['release_speed_c', 'plate_x_bat_flip', 'plate_z', 'intercept_y']
QUANTILE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def fit_quantile_curves(df: pd.DataFrame) -> pd.DataFrame:
    """OOF quantile predictions for every swing with a valid target/features,
    reusing the same 5-fold split as oof_timing_offset.py (fresh KFold here
    since this needs its own fold column -- offset model's own fold
    assignment from Step 1c isn't reused because this is a different model
    entirely, fit on a possibly different valid-row subset)."""
    from sklearn.model_selection import KFold

    needed = OFFSET_FEATURES + ['barrel_distance_v2']
    sub = df.dropna(subset=needed).copy().reset_index(drop=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sub['fold'] = np.nan
    for fold_id, (_, test_idx) in enumerate(kf.split(sub)):
        sub.loc[sub.index[test_idx], 'fold'] = fold_id

    quantile_preds = {q: np.full(len(sub), np.nan) for q in QUANTILE_LEVELS}
    for fold in sorted(sub['fold'].unique()):
        train_mask = sub['fold'] != fold
        test_mask = sub['fold'] == fold
        X_train = sub.loc[train_mask, OFFSET_FEATURES]
        y_train = sub.loc[train_mask, 'barrel_distance_v2']
        X_test = sub.loc[test_mask, OFFSET_FEATURES]

        for q in QUANTILE_LEVELS:
            model = HistGradientBoostingRegressor(
                loss='quantile', quantile=q, max_iter=200, max_depth=6,
                learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)
            quantile_preds[q][test_mask.values] = model.predict(X_test)
        print(f'  fold {fold}: n_test={test_mask.sum():,} done')

    for q in QUANTILE_LEVELS:
        sub[f'q_{q}'] = quantile_preds[q]
    return sub


def integrate_probability(sub: pd.DataFrame, threshold: float) -> np.ndarray:
    """p_i = P(offset_i < threshold) via linear interpolation of each row's
    fitted quantile curve. Quantile levels are sorted with their predicted
    values to enforce monotonicity (independently-fit quantile regressions
    can cross in practice); np.interp clamps outside the fitted range,
    avoiding exact 0/1 probabilities."""
    q_cols = [f'q_{q}' for q in QUANTILE_LEVELS]
    values = sub[q_cols].values  # shape (n, K)
    levels = np.array(QUANTILE_LEVELS)

    p = np.empty(len(sub))
    for i in range(len(sub)):
        row_vals = values[i]
        order = np.argsort(row_vals)  # enforce monotonicity per-row
        p[i] = np.interp(threshold, row_vals[order], levels[order])
    return p


def evaluate_vs_step2(sub: pd.DataFrame, p_option5: np.ndarray) -> dict:
    y = sub['is_contact'].astype(int).values
    p_clipped = np.clip(p_option5, 1e-6, 1 - 1e-6)
    return dict(
        log_loss=log_loss(y, p_clipped),
        brier=brier_score_loss(y, p_clipped),
        misclass_rate=((p_clipped >= 0.5).astype(int) != y).mean(),
        n=len(y),
    )


def plot_calibration(sub: pd.DataFrame, p: np.ndarray, path: str):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(sub['is_contact'].astype(int), np.clip(p, 1e-6, 1-1e-6), n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, 'o-', color=BLUE, label='Option 5 (distributional)')
    ax.plot([0, 1], [0, 1], '--', color=RED, label='perfect calibration')
    ax.set_xlabel('Mean predicted P(contact)'); ax.set_ylabel('Observed contact rate')
    ax.set_title('Option 5 calibration curve', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'-> {path}')


def main():
    df = prepare_data()  # already sets df['is_contact']

    print(f'Threshold C (bat/ball radius sum) = {BARREL_THRESHOLD:.3f} inches')
    sub = fit_quantile_curves(df)
    print(f'\nQuantile curves fit for {len(sub):,} swings')

    p_option5 = integrate_probability(sub, BARREL_THRESHOLD)
    metrics = evaluate_vs_step2(sub, p_option5)
    print(f'\nOption 5 (distributional) held-out-equivalent metrics:')
    print(f'  log_loss={metrics["log_loss"]:.4f}  brier={metrics["brier"]:.4f}  '
          f'misclass_rate={metrics["misclass_rate"]:.4f}  n={metrics["n"]:,}')

    plot_calibration(sub, p_option5, os.path.join(OUT_DIR, 'step3b_option5_calibration.png'))

    eval_path = os.path.join(OUT_DIR, 'evaluation_report.md')
    with open(eval_path, 'a') as f:
        f.write('\n\n## Step 3b -- Option 5 (distributional/threshold-integration model)\n\n')
        f.write(f'Triggered by Step 3 heteroskedasticity check (21.3x whiff / 2.7x contact '
                f'variance ratio, both above the 2-3x threshold).\n\n')
        f.write(f'| model | log_loss | brier | misclass_rate | n |\n|---|---|---|---|---|\n')
        f.write(f'| Option 5 (quantile regression + threshold integration) | '
                f'{metrics["log_loss"]:.4f} | {metrics["brier"]:.4f} | '
                f'{metrics["misclass_rate"]:.4f} | {metrics["n"]:,} |\n\n')
        f.write('Compare against Step 2\'s best classifier (raw_gbm / hybrid_gbm) above -- '
                'per spec 3b, keep whichever wins on held-out log-loss/Brier; prefer the '
                'simpler Step 2 model if roughly tied.\n')
    print(f'-> appended comparison to {eval_path}')

    out_path = os.path.join(OUT_DIR, 'oof_contact_probs_option5.parquet')
    sub[['row_key', 'batter', 'is_contact']].assign(p_contact=p_option5).to_parquet(out_path, index=False)
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
