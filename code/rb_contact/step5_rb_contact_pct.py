"""
step5_rb_contact_pct.py
=========================
RB-Contact% Step 5: compute RB-Contact% from the final chosen model's
out-of-fold probabilities, validate against raw contact% via split-half
reliability (paper Table 2 analog) and a discrimination metric (paper
Table 3 analog, Franks et al. signal/noise decomposition), and apply
Beta-Binomial shrinkage (Step 5.5).

Requires: out/rb_contact/oof_predictions.parquet (Step 1c) AND a saved set
of out-of-fold contact probabilities from the winning Step 2/4 model
(oof_contact_probs.parquet, produced by generate_oof_contact_probs() below --
run once the final model choice is settled).
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR
from model_utils import assemble_design, make_gbm, CONTINUOUS_RAW, BINARY_RAW, X_HYBRID_EXTRA
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial

warnings.filterwarnings('ignore')

MIN_SWINGS = 100  # minimum swings for a batter to be included in validation


def generate_oof_contact_probs(feature_set='hybrid', model_type='gbm') -> pd.DataFrame:
    """
    Produce out-of-fold p_i (final chosen model) for every swing -- same
    5-fold discipline as Step 1c/1b. This is what Step 5's RB-Contact% is
    actually averaging; it must be OOF for the same leakage reasons as
    predicted_timing/predicted_offset.
    """
    df = prepare_data()
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    oof = pd.read_parquet(oof_path)[['row_key', 'fold', 'predicted_int_y', 'predicted_barrel']]
    oof = oof.rename(columns={'predicted_int_y': 'predicted_timing', 'predicted_barrel': 'predicted_offset'})
    df = df.merge(oof, on='row_key', how='inner')

    extra = X_HYBRID_EXTRA if feature_set == 'hybrid' else []
    needed = CONTINUOUS_RAW + BINARY_RAW + extra + ['pitch_type', 'is_contact', 'fold']
    sub = df.dropna(subset=needed).copy()
    pt_categories = sorted(sub['pitch_type'].unique())

    preds = []
    for fold in sorted(sub['fold'].unique()):
        train = sub[sub['fold'] != fold]
        test = sub[sub['fold'] == fold]
        X_train = assemble_design(train, extra, pt_categories)
        X_test = assemble_design(test, extra, pt_categories)
        model = make_gbm()
        model.fit(X_train, train['is_contact'].astype(int))
        p = model.predict_proba(X_test)[:, 1]
        preds.append(pd.DataFrame({'row_key': test['row_key'].values, 'p_contact': p}))
        print(f'  fold {fold}: n={len(test):,}')

    out = pd.concat(preds, ignore_index=True)
    out = sub[['row_key', 'batter', 'game_date', 'is_contact']].merge(out, on='row_key', how='left')
    path = os.path.join(OUT_DIR, 'oof_contact_probs.parquet')
    out.to_parquet(path, index=False)
    print(f'-> {path}')
    return out


def compute_batter_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('batter')
    out = pd.DataFrame({
        'n_swings': g.size(),
        'rb_contact_pct': g['p_contact'].mean(),
        'raw_contact_pct': g['is_contact'].mean(),
    }).reset_index()
    return out[out['n_swings'] >= MIN_SWINGS]


def split_half_reliability(df: pd.DataFrame) -> dict:
    """Split each batter's swings by date (first half of season vs second),
    correlate first-half metric with second-half metric, and compute MAE
    predicting second-half raw contact% from first-half RB vs first-half raw."""
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    median_date = df['game_date'].median()
    first = df[df['game_date'] <= median_date]
    second = df[df['game_date'] > median_date]

    first_m = compute_batter_metrics(first).add_suffix('_h1')
    second_m = compute_batter_metrics(second).add_suffix('_h2')
    m = first_m.merge(second_m, left_on='batter_h1', right_on='batter_h2', how='inner')

    results = {}
    for label, col_h1 in [('RB-Contact%', 'rb_contact_pct_h1'), ('raw contact%', 'raw_contact_pct_h1')]:
        # Split-half correlation of the SAME metric across halves (e.g. H1
        # RB-Contact% vs H2 RB-Contact%), plus MAE using H1 to predict H2's
        # RAW contact% specifically -- the paper's Table 2 comparison point.
        same_col_h2 = col_h1.replace('_h1', '_h2')
        rho, p = stats.spearmanr(m[col_h1], m[same_col_h2])
        mae = np.mean(np.abs(m[col_h1] - m['raw_contact_pct_h2']))
        results[label] = dict(spearman_rho=rho, spearman_p=p,
                              mae_predicting_h2_raw=mae, n_batters=len(m))
    return results, m


def discrimination_metric(df: pd.DataFrame, metric_col: str) -> float:
    """
    Franks et al.-style discrimination: fraction of between-batter variance
    attributable to true skill rather than sampling noise.
      total_var       = variance of per-batter metric across batters
      within_var_mean = average within-batter binomial sampling variance,
                         p_bar*(1-p_bar)/n for each batter (using that
                         batter's own rate and swing count)
      discrimination   = max(0, total_var - within_var_mean) / total_var
    Higher = more of the observed spread reflects real skill differences,
    less is noise. RB-Contact% should score higher than raw contact% if the
    approach is working.
    """
    batter_metrics = compute_batter_metrics(df)
    p_bar = batter_metrics[metric_col]
    n = batter_metrics['n_swings']
    total_var = p_bar.var(ddof=1)
    within_var_mean = (p_bar * (1 - p_bar) / n).mean()
    return max(0.0, total_var - within_var_mean) / total_var


def apply_shrinkage(df: pd.DataFrame, metric_col: str) -> pd.Series:
    batter_metrics = compute_batter_metrics(df)
    alpha, beta = fit_beta_prior_moments(batter_metrics[metric_col].values,
                                         weights=batter_metrics['n_swings'].values)
    shrunk = shrink_beta_binomial(batter_metrics[metric_col].values,
                                  batter_metrics['n_swings'].values, alpha, beta)
    return pd.Series(shrunk, index=batter_metrics['batter'], name=f'{metric_col}_shrunk')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-probs', action='store_true',
                        help='Regenerate oof_contact_probs.parquet (slow -- 5 GBM fits on full data)')
    args = parser.parse_args()

    probs_path = os.path.join(OUT_DIR, 'oof_contact_probs.parquet')
    if args.regen_probs or not os.path.exists(probs_path):
        print('Generating OOF contact probabilities (final model)...')
        df = generate_oof_contact_probs()
    else:
        df = pd.read_parquet(probs_path)
        print(f'Loaded existing {probs_path} ({len(df):,} rows)')

    batter_metrics = compute_batter_metrics(df)
    print(f'\n{len(batter_metrics)} batters with >= {MIN_SWINGS} swings')

    reliability, half_merged = split_half_reliability(df)
    print('\n=== Split-half reliability ===')
    for label, r in reliability.items():
        print(f'  {label}: spearman rho={r["spearman_rho"]:.3f} (p={r["spearman_p"]:.2e}), '
              f'MAE predicting H2 raw contact% = {r["mae_predicting_h2_raw"]:.4f}  (n={r["n_batters"]})')

    disc_rb_val = discrimination_metric(df, 'rb_contact_pct')
    disc_raw_val = discrimination_metric(df, 'raw_contact_pct')
    print(f'\n=== Discrimination (fraction of variance = true skill, not noise) ===')
    print(f'  RB-Contact%:  {disc_rb_val:.4f}')
    print(f'  raw contact%: {disc_raw_val:.4f}')

    shrunk_rb = apply_shrinkage(df, 'rb_contact_pct')
    batter_metrics = batter_metrics.set_index('batter').join(shrunk_rb).reset_index()

    out_path = os.path.join(OUT_DIR, 'rb_contact_pct_by_batter.parquet')
    batter_metrics.to_csv(out_path.replace('.parquet', '.csv'), index=False)
    batter_metrics.to_parquet(out_path, index=False)
    print(f'\n-> {out_path}')

    with open(os.path.join(OUT_DIR, 'validation_report.md'), 'w') as f:
        f.write('# RB-Contact% -- Step 5 Validation Report\n\n')
        f.write('## Split-half reliability\n\n')
        for label, r in reliability.items():
            f.write(f'- **{label}**: Spearman rho={r["spearman_rho"]:.3f} (p={r["spearman_p"]:.2e}), '
                    f'MAE predicting 2nd-half raw contact% = {r["mae_predicting_h2_raw"]:.4f} '
                    f'(n={r["n_batters"]} batters)\n')
        f.write('\n## Discrimination (Franks et al. signal/noise decomposition)\n\n')
        f.write(f'- RB-Contact%: {disc_rb_val:.4f}\n- raw contact%: {disc_raw_val:.4f}\n')
        f.write(f'\nRB-Contact% {"shows higher discrimination (working as intended)" if disc_rb_val > disc_raw_val else "does NOT show higher discrimination -- investigate"}.\n')


if __name__ == '__main__':
    main()
