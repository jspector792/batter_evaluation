"""
step1_2_contact_models.py
===========================
RB-Contact% Steps 1a/1b/2: fit the raw-inputs classifier (Option 2) and,
once out-of-fold predicted_timing/predicted_offset exist (Step 1c), the
hybrid classifier (Option 4). Evaluates every variant via k-fold CV with
swing-level randomization, applies post-hoc calibration to the GBM variants,
and writes out/rb_contact/evaluation_report.md (Step 2).

Fold assignment: if out/rb_contact/oof_predictions.parquet exists, its 'fold'
column is reused verbatim -- this guarantees the hybrid model's own train/test
split never mixes a swing's out-of-fold predicted_timing/predicted_offset
into its own training fold (the leakage the spec's testing checklist calls
out explicitly). If the OOF file doesn't exist yet, only the raw-inputs
model (1a) is built, using a fresh 5-fold split.
"""

import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR
from model_utils import (assemble_design, make_gbm, make_logistic_spline,
                         calibrate, evaluate, CONTINUOUS_RAW, BINARY_RAW,
                         X_HYBRID_EXTRA, interaction_cols)

from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5


def load_folded_data():
    """Swing population + fold assignment. Merges predicted_timing/offset in
    (and confirms the hybrid feature set can be built) iff the OOF file exists."""
    df = prepare_data()
    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    have_hybrid = os.path.exists(oof_path)

    if have_hybrid:
        oof = pd.read_parquet(oof_path)[['row_key', 'fold', 'predicted_int_y', 'predicted_barrel']]
        oof = oof.rename(columns={'predicted_int_y': 'predicted_timing',
                                  'predicted_barrel': 'predicted_offset'})
        df = df.merge(oof, on='row_key', how='inner')
        print(f'Merged OOF predictions ({df["predicted_timing"].notna().sum():,} '
              f'/ {len(df):,} rows have predicted_timing)')
    else:
        print('No oof_predictions.parquet found yet -- building Option 2 (raw-inputs) only.')
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        df['fold'] = np.nan
        for fold_id, (_, test_idx) in enumerate(kf.split(df)):
            df.loc[df.index[test_idx], 'fold'] = fold_id

    return df, have_hybrid


def run_variant(df: pd.DataFrame, feature_set: str, model_type: str) -> dict:
    """
    feature_set: 'raw' or 'hybrid'
    model_type:  'gbm' or 'logistic'
    Returns per-fold metrics (pre- and post-calibration for gbm; logistic has
    no separate calibration step per spec Step 2.3, which calibrates GBM only).
    """
    extra = X_HYBRID_EXTRA if feature_set == 'hybrid' else []
    needed = CONTINUOUS_RAW + BINARY_RAW + extra + ['pitch_type', 'is_contact', 'fold']
    sub = df.dropna(subset=needed).copy()

    # Fixed category list across every fold/split -- a rare pitch type present
    # in one split but not another otherwise produces mismatched dummy columns.
    pt_categories = sorted(sub['pitch_type'].unique())
    pt_cols = [f'pt_{c}' for c in pt_categories]

    fold_metrics = []
    for fold in sorted(sub['fold'].unique()):
        train_full = sub[sub['fold'] != fold]
        test = sub[sub['fold'] == fold]

        # Hold out a calibration split from this fold's training data (Step 2.3) --
        # calibration must be fit on data the GBM itself never trained on.
        train, calib = train_test_split(train_full, test_size=0.15, random_state=SEED,
                                         stratify=train_full['is_contact'])

        X_train = assemble_design(train, extra, pt_categories)
        X_calib = assemble_design(calib, extra, pt_categories)
        X_test  = assemble_design(test, extra, pt_categories)
        y_train, y_calib, y_test = train['is_contact'].astype(int), calib['is_contact'].astype(int), test['is_contact'].astype(int)

        if model_type == 'gbm':
            model = make_gbm()
            model.fit(X_train, y_train)
            metrics_precal = evaluate(model, X_test, y_test)

            cal_model = calibrate(model, X_calib, y_calib, method='isotonic')
            metrics_postcal = evaluate(cal_model, X_test, y_test)
            fold_metrics.append(dict(fold=fold, precal=metrics_precal, postcal=metrics_postcal))
        else:
            model = make_logistic_spline(CONTINUOUS_RAW,
                                         BINARY_RAW + extra + pt_cols + interaction_cols())
            # logistic pipeline expects the passthrough cols to exist verbatim
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            cal_model = None
            fold_metrics.append(dict(fold=fold, precal=metrics, postcal=None))

        # Persist fold 0's fitted model(s) as the representative artifact for
        # this variant (deliverable: models/ -- trained artifacts for every
        # variant). Not a "the" production model -- Step 5/6 train their own
        # full-population models when an unbiased single canonical fit matters.
        if fold == sorted(sub['fold'].unique())[0]:
            import joblib
            models_dir = os.path.join(OUT_DIR, 'models')
            joblib.dump(model, os.path.join(models_dir, f'{feature_set}_{model_type}_fold0.joblib'))
            if cal_model is not None:
                joblib.dump(cal_model, os.path.join(models_dir,
                            f'{feature_set}_{model_type}_fold0_calibrated.joblib'))

        print(f'  [{feature_set}/{model_type}] fold {fold}: '
              f'n_train={len(train):,} n_test={len(test):,} '
              f'log_loss={fold_metrics[-1]["precal"]["log_loss"]:.4f}')

    return fold_metrics


def summarize(fold_metrics: list, key: str) -> dict:
    vals = {m: np.mean([f[key][m] for f in fold_metrics]) for m in ['log_loss', 'brier', 'misclass_rate']}
    stds = {f'{m}_std': np.std([f[key][m] for f in fold_metrics]) for m in ['log_loss', 'brier', 'misclass_rate']}
    return {**vals, **stds, 'n_total': sum(f[key]['n'] for f in fold_metrics)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-n', type=int, default=None,
                        help='Subsample N swings for a quick smoke test')
    args = parser.parse_args()

    df, have_hybrid = load_folded_data()
    if args.sample_n and args.sample_n < len(df):
        # Deliberately NOT reusing SEED here: KFold(random_state=SEED) and
        # DataFrame.sample(random_state=SEED) on the same population size
        # produce correlated permutations (verified: with SEED=42 on this
        # exact row count, .sample() draws land entirely inside fold 0).
        # Production runs never subsample post-fold, so this only matters
        # for smoke tests -- using an unrelated seed avoids it.
        df = df.sample(n=args.sample_n, random_state=SEED + 12345).reset_index(drop=True)
        print(f'Subsampled to {len(df):,} rows for smoke test')

    variants = [('raw', 'logistic'), ('raw', 'gbm')]
    if have_hybrid:
        variants += [('hybrid', 'logistic'), ('hybrid', 'gbm')]

    all_results = {}
    report_rows = []
    for feature_set, model_type in variants:
        label = f'{feature_set}_{model_type}'
        print(f'\n{"="*70}\n{label}\n{"="*70}')
        fold_metrics = run_variant(df, feature_set, model_type)
        all_results[label] = fold_metrics

        precal_summary = summarize(fold_metrics, 'precal')
        report_rows.append(dict(model=label, stage='pre-calibration', **precal_summary))
        if fold_metrics[0]['postcal'] is not None:
            postcal_summary = summarize(fold_metrics, 'postcal')
            report_rows.append(dict(model=label, stage='post-calibration (isotonic)', **postcal_summary))

    report_df = pd.DataFrame(report_rows)
    print('\n' + report_df.to_string(index=False))

    # ── Write evaluation_report.md ────────────────────────────────────────────
    lines = ['# RB-Contact% -- Step 2 Evaluation Report\n']
    lines.append(f'Variants built: {[f"{fs}/{mt}" for fs, mt in variants]}. '
                f'Hybrid variants {"included" if have_hybrid else "NOT YET AVAILABLE (waiting on Step 1c OOF generation)"}.\n')
    lines.append('## Log loss / Brier / misclassification (5-fold CV, mean +/- std across folds)\n')
    rdf = report_df.round(4)
    header = '| ' + ' | '.join(rdf.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(rdf.columns)) + '|'
    body = '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rdf.itertuples(index=False))
    lines.append('\n'.join([header, sep, body]))
    lines.append('\n')
    path = os.path.join(OUT_DIR, 'evaluation_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\n-> {path}')

    # Save raw per-fold results for later steps (calibration curves, Step 4 reuse)
    import pickle
    with open(os.path.join(OUT_DIR, '_step1_2_raw_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
