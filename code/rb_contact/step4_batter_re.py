"""
step4_batter_re.py
====================
RB-Contact% Step 4: does adding batter identity to the contact-probability
model help, and if so, is the improvement real shared signal or overfitting
concentrated in high-swing-count players?

Variant A: base model (feature_set/model_type from Step 2's winner), no
           batter identity.
Variant B: base model + a shrunk target-encoded batter feature -- each
           batter's contact rate, computed from TRAINING folds only, shrunk
           toward the league mean via the Beta-Binomial prior from Step 5.5
           (shrinkage.py), shrinkage strength tied to that batter's training
           swing count. Explicitly NOT a raw one-hot batter ID (spec
           forbids this -- it would overfit high-swing-count players).

Validation: reuses the same swing-level 5-fold split as everywhere else in
this pipeline (a batter's swings ARE split across folds, satisfying the
spec's "grouped... within batter" requirement without needing a separate
split). Reports held-out log-loss/Brier PER swing-count bucket
(<200, 200-500, 500-1000, 1000+) to catch overfitting concentrated in
high-volume players -- the spec's explicit red-flag check.
"""

import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR
from model_utils import (assemble_design, make_gbm, make_logistic_spline,
                         evaluate, CONTINUOUS_RAW, BINARY_RAW, X_HYBRID_EXTRA,
                         interaction_cols)
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial

warnings.filterwarnings('ignore')

SWING_COUNT_BUCKETS = [(0, 200), (200, 500), (500, 1000), (1000, np.inf)]


def load_data(feature_set: str = 'hybrid') -> pd.DataFrame:
    df = prepare_data()
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    if feature_set == 'hybrid':
        if not os.path.exists(oof_path):
            raise FileNotFoundError('oof_predictions.parquet not found -- run Step 1c first, '
                                    'or pass feature_set="raw" to run Step 4 without it.')
        oof = pd.read_parquet(oof_path)[['row_key', 'fold', 'predicted_int_y', 'predicted_barrel']]
        oof = oof.rename(columns={'predicted_int_y': 'predicted_timing',
                                  'predicted_barrel': 'predicted_offset'})
        df = df.merge(oof, on='row_key', how='inner')
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df['fold'] = np.nan
        for fold_id, (_, test_idx) in enumerate(kf.split(df)):
            df.loc[df.index[test_idx], 'fold'] = fold_id
    return df


def add_shrunk_batter_encoding(train: pd.DataFrame, apply_to: pd.DataFrame) -> pd.Series:
    """
    Compute each batter's shrunk contact rate from `train` only, then map it
    onto `apply_to` (which may include rows/batters not in train -- those get
    the league-mean prior, i.e. zero information, exactly as intended).
    """
    batter_stats = train.groupby('batter')['is_contact'].agg(['mean', 'count'])
    alpha, beta = fit_beta_prior_moments(batter_stats['mean'].values,
                                         weights=batter_stats['count'].values)
    batter_stats['shrunk'] = shrink_beta_binomial(
        batter_stats['mean'].values, batter_stats['count'].values, alpha, beta)
    prior_mean = alpha / (alpha + beta)
    mapping = batter_stats['shrunk']
    return apply_to['batter'].map(mapping).fillna(prior_mean)


def bucket_swing_counts(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby('batter')['batter'].transform('size')
    labels = pd.Series('unknown', index=df.index)
    for lo, hi in SWING_COUNT_BUCKETS:
        label = f'{lo}-{hi if hi != np.inf else "+"}'
        labels[(counts >= lo) & (counts < hi)] = label
    return labels


def run_ab_test(df: pd.DataFrame, feature_set: str, model_type: str) -> pd.DataFrame:
    extra = X_HYBRID_EXTRA if feature_set == 'hybrid' else []
    needed = CONTINUOUS_RAW + BINARY_RAW + extra + ['pitch_type', 'is_contact', 'fold', 'batter']
    sub = df.dropna(subset=needed).copy()
    sub['swing_bucket'] = bucket_swing_counts(sub)
    pt_categories = sorted(sub['pitch_type'].unique())

    rows = []
    for fold in sorted(sub['fold'].unique()):
        train = sub[sub['fold'] != fold]
        test = sub[sub['fold'] == fold]

        X_train_a = assemble_design(train, extra, pt_categories)
        X_test_a = assemble_design(test, extra, pt_categories)
        y_train, y_test = train['is_contact'].astype(int), test['is_contact'].astype(int)

        # ── Variant A: no batter identity ───────────────────────────────────
        model_a = make_gbm() if model_type == 'gbm' else make_logistic_spline(
            CONTINUOUS_RAW, BINARY_RAW + extra + [c for c in X_test_a.columns if c.startswith('pt_')] + interaction_cols())
        model_a.fit(X_train_a, y_train)
        p_a = model_a.predict_proba(X_test_a)[:, 1]

        # ── Variant B: + shrunk batter encoding (computed from train only) ──
        train_encoding = add_shrunk_batter_encoding(train, train)
        test_encoding = add_shrunk_batter_encoding(train, test)
        X_train_b = X_train_a.copy(); X_train_b['batter_contact_rate_shrunk'] = train_encoding.values
        X_test_b = X_test_a.copy(); X_test_b['batter_contact_rate_shrunk'] = test_encoding.values

        model_b = make_gbm() if model_type == 'gbm' else make_logistic_spline(
            CONTINUOUS_RAW + ['batter_contact_rate_shrunk'],
            BINARY_RAW + extra + [c for c in X_test_a.columns if c.startswith('pt_')] + interaction_cols())
        model_b.fit(X_train_b, y_train)
        p_b = model_b.predict_proba(X_test_b)[:, 1]

        for variant, p in [('A_no_batter_id', p_a), ('B_shrunk_batter_encoding', p_b)]:
            p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
            test_res = test.assign(p=p_clipped, variant=variant, fold=fold)
            for bucket in test_res['swing_bucket'].unique():
                bres = test_res[test_res['swing_bucket'] == bucket]
                from sklearn.metrics import log_loss, brier_score_loss
                rows.append(dict(
                    fold=fold, variant=variant, swing_bucket=bucket, n=len(bres),
                    log_loss=log_loss(bres['is_contact'].astype(int), bres['p']),
                    brier=brier_score_loss(bres['is_contact'].astype(int), bres['p']),
                ))
        print(f'  fold {fold} done ({feature_set}/{model_type})')

    return pd.DataFrame(rows)


def summarize_and_report(results: pd.DataFrame, path: str):
    summary = (results.groupby(['variant', 'swing_bucket'])
              .apply(lambda g: pd.Series({
                  'log_loss': np.average(g['log_loss'], weights=g['n']),
                  'brier': np.average(g['brier'], weights=g['n']),
                  'n_total': g['n'].sum(),
              }))
              .reset_index())
    print('\n' + summary.to_string(index=False))

    pivot = summary.pivot(index='swing_bucket', columns='variant', values='log_loss')
    if 'A_no_batter_id' in pivot.columns and 'B_shrunk_batter_encoding' in pivot.columns:
        pivot['log_loss_delta_B_minus_A'] = pivot['B_shrunk_batter_encoding'] - pivot['A_no_batter_id']

    lines = ['# RB-Contact% -- Step 4 Batter Random-Effect A/B Test\n']
    lines.append('Positive delta = Variant B (batter encoding) is WORSE (higher log loss) than A.\n')
    lines.append('Red flag per spec: improvement (negative delta) concentrated ONLY in high-swing-count '
                 'buckets suggests overfitting via identity, not genuine shared signal.\n')
    lines.append(pivot.round(4).to_string())
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\n-> {path}')
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-set', choices=['raw', 'hybrid'], default='hybrid')
    parser.add_argument('--model-type', choices=['raw', 'gbm', 'logistic'], default='gbm')
    args = parser.parse_args()

    df = load_data(args.feature_set)
    results = run_ab_test(df, args.feature_set, args.model_type)
    summarize_and_report(results, os.path.join(OUT_DIR, 'step4_batter_re_ab_test.md'))


if __name__ == '__main__':
    main()
